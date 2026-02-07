"""
Explainable AI (xAI) for ConceptCLIP Medical Image Retrieval.

This script demonstrates region-concept alignment visualization:
1. Encodes all test images to create embedding database
2. Given a query image, retrieves top-K similar images
3. Explains WHY retrieval results are relevant by visualizing:
   - Which image regions (patches) align with medical concepts
   - Which concepts are present in query vs retrieved images
   - Spatial heatmaps showing where concepts are detected

Usage:
    # Zero-shot (pretrained model, no fine-tuning)
    python xai_conceptclip.py
    
    # With fine-tuned checkpoint
    python xai_conceptclip.py --checkpoint_path checkpoints/conceptclip_epoch1.pth
    
    # Custom query
    python xai_conceptclip.py --checkpoint_path checkpoint.pth --query_image_id 0a0c223c9feb91155bfb4a101362ffe9
    
    # More retrieval results
    python xai_conceptclip.py --checkpoint_path checkpoint.pth --top_k 10 --num_query_samples 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
import os
from tqdm import tqdm

from model import conceptCLIP
from read_data import VINDRConceptCLIPDataSet


def encode_all_images(model, dataset, device, batch_size=32):
    """Encode all images in dataset to create embedding database.
    
    Returns:
        cls_embeddings: (N, D) normalized CLS embeddings for retrieval
        patch_embeddings: (N, num_patches, D) patch embeddings for explanation
        image_ids: list of image IDs
    """
    model.eval()
    processor = model.processor
    
    cls_embeds = []
    patch_embeds = []
    image_ids = []
    
    print(f"Encoding {len(dataset)} images...")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch_data = [dataset[j] for j in batch_indices]
        
        images = [item['image'] for item in batch_data]
        # Get image IDs from dataset directly using indices
        batch_image_ids = [dataset.image_ids[j] for j in batch_indices]
        
        # Process images
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            # Get both CLS and patch embeddings
            outputs = model.model(pixel_values=pixel_values)
            
            # Extract CLS embeddings (for retrieval)
            if hasattr(outputs, 'image_embeds'):
                cls_embed = outputs.image_embeds  # (B, D)
            else:
                # Fallback: use model.encode_image
                cls_embed = model.encode_image(pixel_values)
            
            # Extract patch embeddings (for explanation)
            if hasattr(outputs, 'last_hidden_state'):
                patch_embed = outputs.last_hidden_state  # (B, N_patches+1, D)
                # Remove CLS token, keep only patches
                patch_embed = patch_embed[:, 1:, :]  # (B, N_patches, D)
            else:
                # Alternative path
                patch_embed = outputs.get('image_features_all', None)
                if patch_embed is not None:
                    patch_embed = patch_embed[:, 1:, :]
            
            cls_embed = F.normalize(cls_embed, dim=-1)
            patch_embed = F.normalize(patch_embed, dim=-1)
            
            cls_embeds.append(cls_embed.cpu())
            patch_embeds.append(patch_embed.cpu())
            image_ids.extend(batch_image_ids)
    
    cls_embeddings = torch.cat(cls_embeds, dim=0)  # (N, D)
    patch_embeddings = torch.cat(patch_embeds, dim=0)  # (N, num_patches, D)
    
    print(f"✓ Encoded {len(image_ids)} images")
    print(f"  CLS shape: {cls_embeddings.shape}, Patch shape: {patch_embeddings.shape}")
    
    return cls_embeddings, patch_embeddings, image_ids


def encode_concepts(model, device):
    """Encode all 22 medical concepts as text embeddings.
    
    Returns:
        concept_embeds: (22, D) normalized concept embeddings
        concept_names: list of 22 concept names
    """
    concept_names = VINDRConceptCLIPDataSet.CONCEPT_COLUMNS
    processor = model.processor
    
    # Format concepts as "a finding of {concept}"
    concept_texts = [f"a finding of {c.lower()}" for c in concept_names]
    
    inputs = processor(text=concept_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_embeds = model.encode_text(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        text_embeds = F.normalize(text_embeds, dim=-1)
    
    return text_embeds.cpu(), concept_names


def compute_patch_concept_attention(patch_embeds, concept_embeds, top_k=5):
    """Compute patch-concept attention scores.
    
    Args:
        patch_embeds: (num_patches, D) normalized patch embeddings
        concept_embeds: (num_concepts, D) normalized concept embeddings
        top_k: number of top concepts to return
    
    Returns:
        attention: (num_patches, num_concepts) attention scores
        top_concepts_idx: (top_k,) indices of most activated concepts
        top_concepts_scores: (top_k,) max attention scores for top concepts
    """
    # Compute similarity matrix: (num_patches, num_concepts)
    attention = torch.mm(patch_embeds, concept_embeds.T)  # cosine similarity
    
    # Find top-K most activated concepts (max pooling over patches)
    max_attention_per_concept = attention.max(dim=0)[0]  # (num_concepts,)
    top_concepts_scores, top_concepts_idx = torch.topk(max_attention_per_concept, top_k)
    
    return attention, top_concepts_idx, top_concepts_scores


def create_attention_heatmap(attention_vector, patch_grid_size=27, image_size=384):
    """Convert patch attention vector to 2D heatmap.
    
    Args:
        attention_vector: (num_patches,) attention scores
        patch_grid_size: grid size (27 for 384x384 image with patch_size=14)
        image_size: output heatmap size
    
    Returns:
        heatmap: (image_size, image_size) heatmap
    """
    # Reshape to 2D grid
    heatmap_2d = attention_vector.reshape(patch_grid_size, patch_grid_size).numpy()
    
    # Resize to image size using PIL
    heatmap_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((image_size, image_size), Image.BILINEAR)
    heatmap = np.array(heatmap_pil) / 255.0
    
    return heatmap


def visualize_retrieval_explanation(query_image, query_id, query_patch_embeds,
                                     retrieved_images, retrieved_ids, retrieved_patch_embeds,
                                     concept_embeds, concept_names, 
                                     similarity_scores, output_path, top_k_concepts=5):
    """Visualize retrieval results with region-concept explanations.
    
    Args:
        query_image: PIL Image
        query_id: str, query image ID
        query_patch_embeds: (num_patches, D)
        retrieved_images: list of PIL Images
        retrieved_ids: list of image IDs
        retrieved_patch_embeds: (K, num_patches, D)
        concept_embeds: (22, D)
        concept_names: list of 22 concept names
        similarity_scores: (K,) retrieval similarity scores
        output_path: save path for visualization
        top_k_concepts: number of concepts to show
    """
    K = len(retrieved_images)
    
    # Compute patch-concept attention for query
    query_attention, query_top_concepts, query_top_scores = compute_patch_concept_attention(
        query_patch_embeds, concept_embeds, top_k=top_k_concepts
    )
    
    # Create figure: 1 query + K retrieved + explanations
    fig = plt.figure(figsize=(20, 4 * (K + 1)))
    
    # --- Query Image ---
    ax_query = plt.subplot(K + 1, 4, 1)
    ax_query.imshow(query_image, cmap='gray')
    ax_query.set_title(f'Query: {query_id[:12]}...', fontsize=12, fontweight='bold')
    ax_query.axis('off')
    
    # Query concepts text
    ax_query_text = plt.subplot(K + 1, 4, 2)
    ax_query_text.axis('off')
    query_text = "Top Detected Concepts:\n"
    for i, (idx, score) in enumerate(zip(query_top_concepts, query_top_scores)):
        query_text += f"{i+1}. {concept_names[idx]}\n   (score: {score:.3f})\n"
    ax_query_text.text(0.1, 0.9, query_text, fontsize=10, verticalalignment='top',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Query heatmaps for top 2 concepts
    for i in range(min(2, top_k_concepts)):
        concept_idx = query_top_concepts[i]
        concept_attention = query_attention[:, concept_idx]
        heatmap = create_attention_heatmap(concept_attention)
        
        ax_heatmap = plt.subplot(K + 1, 4, 3 + i)
        ax_heatmap.imshow(query_image, cmap='gray', alpha=0.7)
        ax_heatmap.imshow(heatmap, cmap='jet', alpha=0.5)
        ax_heatmap.set_title(f'{concept_names[concept_idx]}\n({query_top_scores[i]:.3f})', fontsize=9)
        ax_heatmap.axis('off')
    
    # --- Retrieved Images ---
    for k in range(K):
        row = k + 1
        
        # Retrieved image
        ax_img = plt.subplot(K + 1, 4, row * 4 + 1)
        ax_img.imshow(retrieved_images[k], cmap='gray')
        ax_img.set_title(f'#{k+1}: {retrieved_ids[k][:12]}...\nSim: {similarity_scores[k]:.4f}', 
                         fontsize=10)
        ax_img.axis('off')
        
        # Compute attention for this retrieved image
        retr_attention, retr_top_concepts, retr_top_scores = compute_patch_concept_attention(
            retrieved_patch_embeds[k], concept_embeds, top_k=top_k_concepts
        )
        
        # Concept text
        ax_text = plt.subplot(K + 1, 4, row * 4 + 2)
        ax_text.axis('off')
        retr_text = "Concepts:\n"
        for i, (idx, score) in enumerate(zip(retr_top_concepts, retr_top_scores)):
            # Highlight if concept also in query top-K
            marker = "★" if idx in query_top_concepts else " "
            retr_text += f"{marker} {i+1}. {concept_names[idx]}\n    ({score:.3f})\n"
        ax_text.text(0.1, 0.9, retr_text, fontsize=8, verticalalignment='top',
                     fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Heatmaps for top 2 concepts
        for i in range(min(2, top_k_concepts)):
            concept_idx = retr_top_concepts[i]
            concept_attention = retr_attention[:, concept_idx]
            heatmap = create_attention_heatmap(concept_attention)
            
            ax_heatmap = plt.subplot(K + 1, 4, row * 4 + 3 + i)
            ax_heatmap.imshow(retrieved_images[k], cmap='gray', alpha=0.7)
            ax_heatmap.imshow(heatmap, cmap='jet', alpha=0.5)
            ax_heatmap.set_title(f'{concept_names[concept_idx]}\n({retr_top_scores[i]:.3f})', fontsize=8)
            ax_heatmap.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def generate_text_explanation(query_id, retrieved_ids, similarity_scores, 
                               query_top_concepts, retrieved_top_concepts_list,
                               concept_names):
    """Generate natural language explanation for retrieval results.
    
    Returns:
        explanation: str, human-readable explanation
    """
    explanation = f"Retrieval Explanation for Query: {query_id}\n"
    explanation += "=" * 70 + "\n\n"
    
    # Query concepts
    explanation += "Query Image Key Findings:\n"
    for i, idx in enumerate(query_top_concepts[:5]):
        explanation += f"  {i+1}. {concept_names[idx]}\n"
    explanation += "\n"
    
    # Retrieved images and alignment
    explanation += "Why These Images Were Retrieved:\n"
    explanation += "-" * 70 + "\n"
    
    for k, (img_id, sim, retr_concepts) in enumerate(zip(retrieved_ids, similarity_scores, 
                                                          retrieved_top_concepts_list)):
        explanation += f"\n#{k+1} (Similarity: {sim:.4f}) - {img_id}\n"
        
        # Find overlapping concepts
        overlap = set(query_top_concepts.tolist()[:5]) & set(retr_concepts.tolist()[:5])
        overlap_names = [concept_names[idx] for idx in overlap]
        
        if len(overlap_names) > 0:
            explanation += f"  ✓ Shared findings: {', '.join(overlap_names)}\n"
        else:
            explanation += f"  • Similar visual features with different concept distribution\n"
        
        explanation += f"  • Top findings: {', '.join([concept_names[i] for i in retr_concepts[:3]])}\n"
    
    explanation += "\n" + "=" * 70 + "\n"
    
    return explanation


def parse_args():
    parser = argparse.ArgumentParser(description='ConceptCLIP xAI for Medical Image Retrieval')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained ConceptCLIP checkpoint (optional, uses pretrained model if not provided)')
    parser.add_argument('--csv_file', type=str, 
                        default='vindr/image_labels_test.csv',
                        help='Path to test CSV')
    parser.add_argument('--data_dir', type=str,
                        default='D:/VinDR-CXR-dataset/vinbigdata-chest-xray-original-png/test/test',
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='xai_outputs',
                        help='Directory to save visualizations')
    parser.add_argument('--query_image_id', type=str, default=None,
                        help='Specific image ID to use as query (random if not specified)')
    parser.add_argument('--num_query_samples', type=int, default=3,
                        help='Number of random queries to test')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of retrieved images to show')
    parser.add_argument('--top_k_concepts', type=int, default=5,
                        help='Number of top concepts to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ConceptCLIP Explainable Medical Image Retrieval")
    print("=" * 70)
    
    # 1. Load model
    print("\n[1/5] Loading ConceptCLIP model...")
    model = conceptCLIP(
        model_name='JerrryNie/ConceptCLIP',
        unfreeze_vision_layers=4,
        unfreeze_text_layers=2
    )
    
    if args.checkpoint_path:
        print(f"   Loading fine-tuned checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded checkpoint")
    else:
        print("   Using pretrained ConceptCLIP model (zero-shot, no fine-tuning)")
    
    model = model.to(device)
    model.eval()
    print("   ✓ Model loaded")
    
    # 2. Load dataset
    print("\n[2/5] Loading VinDR test dataset...")
    dataset = VINDRConceptCLIPDataSet(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        return_pil=True
    )
    print(f"   ✓ Loaded {len(dataset)} test images")
    
    # 3. Encode all images
    print("\n[3/5] Encoding all images (this may take a few minutes)...")
    cls_embeddings, patch_embeddings, image_ids = encode_all_images(
        model, dataset, device, batch_size=args.batch_size
    )
    
    # 4. Encode concepts
    print("\n[4/5] Encoding medical concepts...")
    concept_embeds, concept_names = encode_concepts(model, device)
    print(f"   ✓ Encoded {len(concept_names)} concepts")
    
    # 5. Run retrieval and explanation
    print(f"\n[5/5] Running retrieval with explanation...")
    
    # Select query images
    if args.query_image_id:
        query_indices = [image_ids.index(args.query_image_id)]
    else:
        query_indices = np.random.choice(len(dataset), args.num_query_samples, replace=False)
    
    print(f"   Testing {len(query_indices)} query images...")
    
    for query_idx in query_indices:
        query_id = image_ids[query_idx]
        query_cls_embed = cls_embeddings[query_idx:query_idx+1]  # (1, D)
        query_patch_embed = patch_embeddings[query_idx]  # (num_patches, D)
        query_data = dataset[query_idx]
        query_image = query_data['image']
        
        print(f"\n   Query: {query_id}")
        
        # Compute retrieval similarity (exclude query itself)
        similarities = torch.mm(query_cls_embed, cls_embeddings.T).squeeze()  # (N,)
        similarities[query_idx] = -1  # Exclude self
        
        # Get top-K
        top_k_scores, top_k_indices = torch.topk(similarities, args.top_k)
        
        # Gather retrieved data
        retrieved_images = []
        retrieved_ids = []
        retrieved_patch_embeds = []
        
        for idx in top_k_indices:
            retr_data = dataset[idx]
            retrieved_images.append(retr_data['image'])
            retrieved_ids.append(image_ids[idx])
            retrieved_patch_embeds.append(patch_embeddings[idx])
        
        retrieved_patch_embeds = torch.stack(retrieved_patch_embeds)  # (K, num_patches, D)
        
        # Compute top concepts
        _, query_top_concepts, _ = compute_patch_concept_attention(
            query_patch_embed, concept_embeds, top_k=args.top_k_concepts
        )
        
        retrieved_top_concepts_list = []
        for retr_patch in retrieved_patch_embeds:
            _, top_concepts, _ = compute_patch_concept_attention(
                retr_patch, concept_embeds, top_k=args.top_k_concepts
            )
            retrieved_top_concepts_list.append(top_concepts)
        
        # Generate text explanation
        explanation = generate_text_explanation(
            query_id, retrieved_ids, top_k_scores.numpy(),
            query_top_concepts, retrieved_top_concepts_list, concept_names
        )
        
        # Save text explanation
        text_path = os.path.join(args.output_dir, f'explanation_{query_id}.txt')
        with open(text_path, 'w') as f:
            f.write(explanation)
        print(f"   ✓ Saved text explanation: {text_path}")
        
        # Visualize
        vis_path = os.path.join(args.output_dir, f'retrieval_{query_id}.png')
        visualize_retrieval_explanation(
            query_image, query_id, query_patch_embed,
            retrieved_images, retrieved_ids, retrieved_patch_embeds,
            concept_embeds, concept_names,
            top_k_scores.numpy(), vis_path, 
            top_k_concepts=args.top_k_concepts
        )
        
        # Print summary
        print(f"\n   {explanation}")
    
    print("\n" + "=" * 70)
    print(f"✓ Completed! Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
