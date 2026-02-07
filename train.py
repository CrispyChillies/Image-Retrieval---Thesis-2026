import os
import random

import torch
from torch.optim import Adam, AdamW
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet, VINDRDataSet, VINDRConceptCLIPDataSet
from loss import TripletMarginLoss, WeightedMultiLabelTripletLoss, ConceptCLIPLoss
from sampler import PKSampler

from model import ResNet50, DenseNet121, ConvNeXtV2, SwinV2, conceptCLIP

from sklearn.metrics import average_precision_score
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq, rank=0, lambda_area=0.1, lambda_sparse=0.01):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].to(device), data[1].to(device)

        output = model(samples)
        if isinstance(output, tuple) and len(output) == 2:
            embeddings, attn = output
            has_attention = True
        else:
            embeddings = output
            has_attention = False

        # Metric Loss
        loss, frac_pos_triplets = criterion(embeddings, targets)
        
        # Attention Loss (only if model supports it)
        if has_attention:
            loss_area = attn.mean()
            loss_sparse = torch.mean(attn * torch.log(attn + 1e-8))
            loss = loss + lambda_area * loss_area + lambda_sparse * loss_sparse

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            if rank == 0:
                print('[{:d}, {:d}] | loss: {:.4f} | % avg hard triplets: {:.2f}%'.format(
                    epoch, i, avg_loss, avg_trip))
            running_loss = 0
            running_frac_pos_triplets = 0


# ============================================================================
# ConceptCLIP Training: IT-Align + RC-Align on VinDR
# ============================================================================

def conceptclip_collate_fn(batch):
    """Custom collate for VINDRConceptCLIPDataSet dicts with PIL images."""
    images = [item['image'] for item in batch]       # list of PIL Images
    texts = [item['text'] for item in batch]          # list of strings
    concept_names = [item['concept_names'] for item in batch]  # list of list[str]
    concept_labels = torch.stack([item['concept_labels'] for item in batch])
    disease_labels = torch.stack([item['disease_labels'] for item in batch])
    all_labels = torch.stack([item['all_labels'] for item in batch])
    
    return {
        'images': images,
        'texts': texts,
        'concept_names': concept_names,
        'concept_labels': concept_labels,
        'disease_labels': disease_labels,
        'all_labels': all_labels,
    }


def encode_concepts_for_rc_align(model_without_ddp, concept_names_batch, device):
    """Encode individual concept names into embeddings for RC-Align.
    
    For each sample in the batch, encode its active concepts separately to get
    per-concept embeddings. Uses ConceptCLIP's text encoder.
    
    Args:
        model_without_ddp: unwrapped conceptCLIP model
        concept_names_batch: list of B lists, each containing concept name strings
        device: torch device
    
    Returns:
        concept_embeds_list: list of B tensors, each (w_i, D) or None if no concepts
    """
    processor = model_without_ddp.processor
    concept_embeds_list = []
    
    for concept_names in concept_names_batch:
        if len(concept_names) == 0:
            concept_embeds_list.append(None)
            continue
        
        # Encode each concept as a short text prompt
        concept_texts = [f"a finding of {c.lower()}" for c in concept_names]
        
        # Tokenize all concepts for this sample
        text_inputs = processor(
            text=concept_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items() 
                       if k in ['input_ids', 'attention_mask']}
        
        concept_features = model_without_ddp.encode_text(**text_inputs)  # (w, D)
        
        concept_embeds_list.append(concept_features)
    
    return concept_embeds_list


def train_epoch_conceptclip(model, optimizer, criterion, data_loader, device, epoch,
                            print_freq, rank=0, scaler=None):
    """Training loop for ConceptCLIP with IT-Align + RC-Align.
    
    Args:
        model: conceptCLIP model (possibly DDP-wrapped)
        optimizer: AdamW optimizer
        criterion: ConceptCLIPLoss instance
        data_loader: DataLoader yielding dicts from VINDRConceptCLIPDataSet
        device: torch device
        epoch: current epoch number
        print_freq: logging frequency
        rank: DDP rank
        scaler: optional GradScaler for mixed precision
    """
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    processor = model_without_ddp.processor
    
    running_loss = 0
    running_it_loss = 0
    running_rc_loss = 0
    
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        
        images = batch['images']          # list of PIL Images
        texts = batch['texts']            # list of strings
        concept_names = batch['concept_names']  # list of list[str]
        
        # Process images and text through ConceptCLIP processor
        # The processor handles: resize to 384x384 (patch_size=14 → 27×27=729 patches),
        # normalize with SigLIP stats, tokenize text with PubMedBERT tokenizer
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Separate image and text inputs
        pixel_values = inputs.get('pixel_values')
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        # Mixed precision forward pass
        use_amp = scaler is not None
        with autocast('cuda', enabled=use_amp):
            # Forward pass: get all features
            outputs = model_without_ddp.forward_clip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            image_features = outputs['image_features']              # (B, D)
            text_features = outputs['text_features']                # (B, D)
            image_token_features = outputs['image_token_features']  # (B, N_patches, D)
            logit_scale = outputs['logit_scale']
            logit_bias = outputs.get('logit_bias', None)
            
            # Encode individual concept embeddings for RC-Align
            concept_embeds_list = encode_concepts_for_rc_align(
                model_without_ddp, concept_names, device
            )
            
            # Compute combined loss
            total_loss, it_loss, rc_loss = criterion(
                image_features=image_features,
                text_features=text_features,
                image_token_features=image_token_features,
                concept_text_features_list=concept_embeds_list,
                logit_scale=logit_scale,
                logit_bias=logit_bias
            )
        
        # Backward pass
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        running_loss += total_loss.item()
        running_it_loss += it_loss.item()
        running_rc_loss += rc_loss.item()
        
        if i % print_freq == print_freq - 1:
            n = i + 1
            avg_loss = running_loss / print_freq
            avg_it = running_it_loss / print_freq
            avg_rc = running_rc_loss / print_freq
            if rank == 0:
                print(f'[{epoch}, {n}] | total: {avg_loss:.4f} | '
                      f'IT-Align: {avg_it:.4f} | RC-Align: {avg_rc:.4f}')
            running_loss = 0
            running_it_loss = 0
            running_rc_loss = 0


@torch.no_grad()
def evaluate_conceptclip(model, loader, device, rank=0, world_size=1):
    """Evaluate ConceptCLIP: image-image retrieval using CLS embeddings + multi-label mAP.
    
    Same evaluation as standard pipeline but handles dict-based dataloader output
    and uses the ConceptCLIP processor for image encoding.
    """
    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    processor = model_without_ddp.processor
    
    embeds, labels = [], []
    
    for batch in loader:
        images = batch['images']           # list of PIL Images
        all_labels = batch['all_labels']   # (B, 28)
        
        # Process images through ConceptCLIP processor
        img_inputs = processor(images=images, return_tensors="pt")
        pixel_values = img_inputs['pixel_values'].to(device)
        
        # Get image embeddings via standard forward (returns normalized CLS)
        embedding = model_without_ddp(pixel_values)  # (B, D)
        embeds.append(embedding.cpu())
        labels.append(all_labels)
    
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Gather from all processes if DDP
    if world_size > 1:
        embeds = embeds.to(device)
        labels = labels.to(device)
        embeds_list = [torch.zeros_like(embeds) for _ in range(world_size)]
        labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
        dist.all_gather(embeds_list, embeds)
        dist.all_gather(labels_list, labels)
        embeds = torch.cat(embeds_list, dim=0).cpu()
        labels = torch.cat(labels_list, dim=0).cpu()
    
    # Multi-label mAP evaluation (same Jaccard-based approach)
    embeds_norm = F.normalize(embeds, p=2, dim=1)
    sim_matrix = torch.mm(embeds_norm, embeds_norm.t())
    sim_matrix.fill_diagonal_(-1)
    
    aps = []
    for i in range(len(embeds)):
        intersect = (labels[i] * labels).sum(dim=1)
        union = (labels[i] + labels).clamp(max=1).sum(dim=1)
        jaccard = intersect / (union + 1e-8)
        
        binary_relevance = (jaccard > 0.4).float().numpy()
        
        if np.sum(binary_relevance) > 0:
            ap = average_precision_score(binary_relevance, sim_matrix[i].numpy())
            aps.append(ap)
    
    mean_ap = np.mean(aps) * 100 if len(aps) > 0 else 0.0
    if rank == 0:
        print(f'>> ConceptCLIP mAP (28 labels, Jaccard>0.4): {mean_ap:.3f}%')
    return mean_ap


def retrieval_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu()
        target = target.cpu()
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res


@torch.no_grad()
def evaluate(model, loader, device, rank=0, world_size=1):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples = data[0].to(device)
        _labels = data[1].to(device)
        out = model(samples)
        
        # If model returns a tuple (embeddings, attention), use only embeddings
        embedding = out[0] if isinstance(out, tuple) else out
        embeds.append(embedding.cpu())
        labels.append(_labels.cpu())

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    # Gather embeddings and labels from all processes if using DDP
    if world_size > 1:
        embeds = embeds.to(device)
        labels = labels.to(device)
        embeds_list = [torch.zeros_like(embeds) for _ in range(world_size)]
        labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
        dist.all_gather(embeds_list, embeds)
        dist.all_gather(labels_list, labels)
        embeds = torch.cat(embeds_list, dim=0).cpu()
        labels = torch.cat(labels_list, dim=0).cpu()

    # Check if multi-label dataset (labels have multiple dimensions)
    if labels.dim() > 1 and labels.size(1) > 1:
        # Multi-label mAP evaluation
        embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)
        sim_matrix = torch.mm(embeds_norm, embeds_norm.t())
        sim_matrix.fill_diagonal_(-1)

        aps = []
        for i in range(len(embeds)):
            intersect = (labels[i] * labels).sum(dim=1)
            union = (labels[i] + labels).clamp(max=1).sum(dim=1)
            jaccard = intersect / (union + 1e-8)

            binary_relevance = (jaccard > 0.4).float().numpy()
            
            if np.sum(binary_relevance) > 0:
                ap = average_precision_score(binary_relevance, sim_matrix[i].numpy())
                aps.append(ap)

        mean_ap = np.mean(aps) * 100
        if rank == 0:
            print(f'>> Mean Average Precision (mAP): {mean_ap:.3f}%')
        return mean_ap
    else:
        # Single-label R@1 evaluation
        dists = -torch.cdist(embeds, embeds)
        dists.fill_diagonal_(torch.tensor(float('-inf')))

        accuracy = retrieval_accuracy(dists, labels)[0].item()
        if rank == 0:
            print('>> R@1 accuracy: {:.3f}%'.format(accuracy))
        return accuracy


def save(model, epoch, save_dir, args, is_best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = args.dataset+'_'+args.model
    if args.embedding_dim:
        file_name += '_embed_'+str(args.embedding_dim)
    if args.anomaly:
        file_name += '_anomaly'
    if args.rand_resize:
        file_name += '_randresize'
    file_name += '_seed_'+str(args.seed)
    
    if is_best:
        file_name += '_best_ckpt.pth'
    else:
        file_name += '_epoch_'+str(epoch)+'_ckpt.pth'

    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f'>> Checkpoint saved: {save_path}')
    return save_path


def main(args):
    # Setup DDP if enabled
    rank = 0
    world_size = 1
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        if rank == 0:
            print(f"Using DDP with {world_size} GPUs")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    p = args.labels_per_batch if not args.anomaly else args.labels_per_batch - 1
    k = args.samples_per_label
    batch_size = p * k
    
    # For DDP, keep same batch size per GPU (effective batch = batch_size * world_size)
    # This maximizes GPU utilization and training speed
    per_gpu_batch_size = batch_size
    if rank == 0 and args.use_ddp:
        print(f"Per-GPU batch size: {per_gpu_batch_size}, Effective batch size: {per_gpu_batch_size * world_size}")

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    elif args.model == 'convnextv2':
        model = ConvNeXtV2(embedding_dim=args.embedding_dim)
    elif args.model == 'swinv2':
        model = SwinV2(embedding_dim=args.embedding_dim)
    elif args.model == 'hybrid_convnext_vit':
        model = HybridConvNeXtViT(embedding_dim=args.embedding_dim)
    elif args.model == 'conceptclip':
        model = conceptCLIP(
            embedding_dim=args.embedding_dim,
            unfreeze_vision_layers=args.unfreeze_vision_layers,
            unfreeze_text_layers=args.unfreeze_text_layers,
        )
    elif args.model == 'resnet50_attention':
        model = Resnet50_with_Attention(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError('Model not supported!')

    if os.path.isfile(args.resume):
        if rank == 0:
            print("=> loading checkpoint")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
        if rank == 0:
            print("=> loaded checkpoint")
    else:
        if rank == 0:
            print("=> no checkpoint found")

    model.to(device)
    
    # Wrap model with DDP if enabled
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, 
                    find_unused_parameters=False, 
                    gradient_as_bucket_view=False)  # Avoids stride mismatch warning

    # Use ConceptCLIPLoss for ConceptCLIP + VinDR, WeightedMultiLabel for other vindr, else standard TripletMarginLoss
    if args.model == 'conceptclip' and args.dataset == 'vindr':
        criterion = ConceptCLIPLoss(alpha=args.rc_alpha)
    elif args.dataset == 'vindr':
        criterion = WeightedMultiLabelTripletLoss(margin=args.margin)
    else:
        criterion = TripletMarginLoss(margin=args.margin)
    
    # Setup optimizer with different learning rates for different model parts
    if args.model == 'conceptclip':
        # ConceptCLIP-specific optimizer: AdamW with differential LR for backbone vs head
        model_without_ddp = model.module if args.use_ddp else model
        
        # Collect only trainable parameters (requires_grad=True)
        backbone_params = []
        head_params = []
        
        for name, param in model_without_ddp.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        trainable_count = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model_without_ddp.parameters())
        if rank == 0:
            print(f"ConceptCLIP: {trainable_count:,} / {total_count:,} parameters trainable "
                  f"({100*trainable_count/total_count:.1f}%)")
        
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': args.lr * 0.1})
        if head_params:
            param_groups.append({'params': head_params, 'lr': args.lr})
        if not param_groups:
            # Fallback: train everything
            param_groups = [{'params': model_without_ddp.parameters(), 'lr': args.lr}]
        
        optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    elif hasattr(model.module if args.use_ddp else model, 'f1') and hasattr(model.module if args.use_ddp else model, 'f2') and hasattr(model.module if args.use_ddp else model, 'attention'):
        # Attention-based model with custom parameter groups
        model_without_ddp = model.module if args.use_ddp else model
        optimizer = Adam([
            {'params': model_without_ddp.f1.parameters(), 'lr': args.lr * 0.1},
            {'params': model_without_ddp.f2.parameters(), 'lr': args.lr * 0.1},
            {'params': model_without_ddp.attention.parameters(), 'lr': args.lr * 0.01},
            {'params': model_without_ddp.bnneck.parameters(), 'lr': args.lr},
            {'params': model_without_ddp.fc.parameters(), 'lr': args.lr},
        ])
    elif args.model in ['convnextv2', 'hybrid_convnext_vit']:
        # ConvNeXt or Hybrid model - use different LR for backbone vs head
        model_without_ddp = model.module if args.use_ddp else model
        backbone_params = []
        head_params = []
        
        for name, param in model_without_ddp.named_parameters():
            if 'fc' in name or 'fusion' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
            {'params': head_params, 'lr': args.lr}  # Higher LR for new head
        ])
    else:
        # Simple optimizer for other models
        optimizer = Adam(model.parameters(), lr=args.lr)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # Use 384x384 for ConvNeXtV2, SwinV2 and Hybrid models, 224x224 for other models
    img_size = 384 if args.model in ['convnextv2', 'swinv2', 'hybrid_convnext_vit'] else 224
    resize_size = 432 if img_size == 384 else 256

    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(resize_size),
        transforms.RandomCrop(img_size, padding=4) if args.rand_resize else transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # Set up dataset and dataloader for embedding learning using triplet loss
    if args.dataset == 'covid':
        train_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'train'),
                                         image_list_file=args.train_image_list,
                                         use_covid=not args.anomaly,  # whether or not to use COVID in training
                                         mask_dir=os.path.join(
                                             args.mask_dir, 'train') if args.mask_dir else None,
                                         transform=train_transform)
        test_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                        image_list_file=args.test_image_list,
                                        mask_dir=os.path.join(
                                            args.mask_dir, 'test') if args.mask_dir else None,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        train_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
                                    image_list_file=args.train_image_list,
                                    use_melanoma=not args.anomaly,  # whether or not to use melanoma in training
                                    mask_dir=os.path.join(
                                        args.mask_dir, 'train') if args.mask_dir else None,
                                    transform=train_transform)
        test_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
                                   image_list_file=args.test_image_list,
                                   mask_dir=os.path.join(
                                       args.mask_dir, 'test') if args.mask_dir else None,
                                   transform=test_transform)
    elif args.dataset == 'tbx11k':
        train_dataset = TBX11kDataSet(data_dir=os.path.join(args.dataset_dir, 'train'),
                                      csv_file=args.train_image_list,
                                      transform=train_transform)
        test_dataset = TBX11kDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                     csv_file=args.test_image_list,
                                     transform=test_transform)
    elif args.dataset == 'vindr':
        if args.model == 'conceptclip':
            # Use ConceptCLIP-specific dataset (returns PIL images + concept-rich text)
            train_dataset = VINDRConceptCLIPDataSet(
                data_dir=os.path.join(args.dataset_dir, 'train_data/train'),
                csv_file=args.train_image_list,
                return_pil=True,
            )
            test_dataset = VINDRConceptCLIPDataSet(
                data_dir=os.path.join(args.dataset_dir, 'test_data/test'),
                csv_file=args.test_image_list,
                return_pil=True,
            )
        else:
            train_dataset = VINDRDataSet(data_dir=os.path.join(args.dataset_dir, 'train_data/train'),
                                         csv_file=args.train_image_list,
                                         transform=train_transform)
            test_dataset = VINDRDataSet(data_dir=os.path.join(args.dataset_dir, 'test_data/test'),
                                        csv_file=args.test_image_list,
                                        transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    # targets is a list where the i_th element corresponds to the label of i_th dataset element.
    # This is required for PKSampler to randomly sample from exactly p classes. You will need to
    # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
    # targets attribute with the same format.
    targets = train_dataset.labels

    # Override batch_size if explicitly provided
    if args.batch_size:
        batch_size = args.batch_size
        per_gpu_batch_size = batch_size
    
    # Determine if we need special collate for ConceptCLIP
    use_conceptclip_collate = (args.model == 'conceptclip' and args.dataset == 'vindr')
    collate_fn = conceptclip_collate_fn if use_conceptclip_collate else None

    # Setup samplers and dataloaders
    if args.use_ddp:
        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True if not use_conceptclip_collate else False,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=True if args.workers > 0 else False,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True if not use_conceptclip_collate else False,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=True if args.workers > 0 else False,
            collate_fn=collate_fn
        )
    else:
        # Use RandomSampler or PKSampler based on use_random_sampler flag
        if args.use_random_sampler or use_conceptclip_collate:
            # ConceptCLIP always uses RandomSampler (no PKSampler for CLIP training)
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = PKSampler(targets, p, k)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

    # Mixed precision scaler for ConceptCLIP
    scaler = GradScaler('cuda') if (use_conceptclip_collate and args.amp) else None
    if scaler is not None and rank == 0:
        print('Using mixed precision training (AMP)')

    # Track best model
    best_metric = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f'\n{"="*60}')
            print(f'Training epoch {epoch}/{args.epochs}...')
            print(f'{"="*60}')
        
        if use_conceptclip_collate:
            train_epoch_conceptclip(model, optimizer, criterion, train_loader,
                                    device, epoch, args.print_freq, rank=rank,
                                    scaler=scaler)
        else:
            train_epoch(model, optimizer, criterion, train_loader,
                        device, epoch, args.print_freq, rank=rank)
        
        # Evaluate every N epochs
        eval_freq = args.eval_freq if hasattr(args, 'eval_freq') else 2
        if epoch % eval_freq == 0:
            # Clear CUDA cache before evaluation to avoid OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Synchronize all processes before evaluation
            if args.use_ddp:
                dist.barrier()
            
            if rank == 0:
                print(f'\n{"="*60}')
                print(f'Evaluating epoch {epoch}...')
                print(f'{"="*60}')
            
            if use_conceptclip_collate:
                current_metric = evaluate_conceptclip(model, test_loader, device, 
                                                       rank=rank, world_size=world_size)
            else:
                current_metric = evaluate(model, test_loader, device, rank=rank, world_size=world_size)
            
            # Save best model (only from rank 0)
            if rank == 0:
                model_to_save = model.module if args.use_ddp else model
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    print(f'\n🎯 New best model! Metric: {current_metric:.3f}% (epoch {epoch})')
                    save(model_to_save, epoch, args.save_dir, args, is_best=True)
                else:
                    print(f'\nCurrent: {current_metric:.3f}%, Best: {best_metric:.3f}% (epoch {best_epoch})')
                
                # Also save periodic checkpoint every 10 epochs
                if epoch % 10 == 0:
                    save(model_to_save, epoch, args.save_dir, args, is_best=False)
            
            # Clear cache after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if rank == 0:
        print(f'\n{"="*60}')
        print('Training completed!')
        print(f'{"="*60}')
        print(f'Best model: Epoch {best_epoch} with metric: {best_metric:.3f}%')
        print(f'Best model saved in: {args.save_dir}')
        print(f'{"="*60}')
    
    # Cleanup DDP
    if args.use_ddp:
        dist.destroy_process_group()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid, isic, tbx11k, or vindr)')
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data/',
                        help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--rand-resize', action='store_true',
                        help='Use random resizing data augmentation')
    parser.add_argument('--anomaly', action='store_true',
                        help='Train without anomaly class')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121, resnet50, convnextv2, swinv2, hybrid_convnext_vit, conceptclip, or resnet50_attention)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone weights (useful for ConceptCLIP)')
    parser.add_argument('-p', '--labels-per-batch', default=3, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=16, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='Number of training epochs to run')
    parser.add_argument('--eval-freq', default=2, type=int,
                        help='Evaluate model every N epochs')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--lr', default=0.0001,
                        type=float, help='Learning rate')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Triplet loss margin')
    parser.add_argument('--print-freq', default=5,
                        type=int, help='Print frequency')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='Model save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='Batch size for training (overrides p*k calculation)')
    parser.add_argument('--use-random-sampler', action='store_true',
                        help='Use RandomSampler instead of PKSampler (for non-DDP training)')
    parser.add_argument('--use-ddp', action='store_true',
                        help='Use Distributed Data Parallel training')
    
    # ConceptCLIP-specific arguments
    parser.add_argument('--rc-alpha', default=0.5, type=float,
                        help='Weight for RC-Align loss (paper default: 0.5)')
    parser.add_argument('--unfreeze-vision-layers', default=4, type=int,
                        help='Number of vision encoder layers to unfreeze from the top')
    parser.add_argument('--unfreeze-text-layers', default=2, type=int,
                        help='Number of text encoder layers to unfreeze from the top')
    parser.add_argument('--weight-decay', default=0.01, type=float,
                        help='Weight decay for AdamW optimizer (ConceptCLIP)')
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision training (recommended for ConceptCLIP)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
