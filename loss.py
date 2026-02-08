'''
    Pytorch adaptation of https://omoindrot.github.io/triplet-loss
    https://github.com/omoindrot/tensorflow-triplet-loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2., mining='batch_all'):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.mining = mining

        if mining == 'batch_all':
            self.loss_fn = batch_all_triplet_loss
        if mining == 'batch_hard':
            self.loss_fn = batch_hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)


def batch_hard_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # hardest positive for every anchor
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # Add max value in each row to invalid negatives
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # hardest negative for every anchor
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss[triplet_loss < 0] = 0

    triplet_loss = triplet_loss.mean()

    return triplet_loss, -1


def batch_all_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return labels.unsqueeze(0) != labels.unsqueeze(1)

class WeightedMultiLabelTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(WeightedMultiLabelTripletLoss, self).__init__()
        self.margin = margin

    def compute_jaccard_sim(self, labels):
        """
        Tính toán ma trận Jaccard Similarity cho toàn bộ batch.
        labels shape: (batch_size, num_classes)
        """
        # Giao (Intersection): sum of element-wise min
        intersection = torch.matmul(labels, labels.t())
        
        # Tổng số nhãn của mỗi ảnh
        label_sums = labels.sum(dim=1).view(-1, 1)
        
        # Hợp (Union): |A| + |B| - |A \cap B|
        union = label_sums + label_sums.t() - intersection
        
        # Jaccard similarity
        jaccard = intersection / (union + 1e-8)
        return jaccard

    def forward(self, embeddings, labels):
        # 1. Chuẩn hóa embedding về unit sphere (giúp tính khoảng cách ổn định hơn)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Tính ma trận khoảng cách Pairwise Euclidean
        # dist(i,j) = sqrt(2 - 2 * cos_sim) khi đã normalize L2
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 3. Tính ma trận Jaccard Similarity
        jaccard_matrix = self.compute_jaccard_sim(labels)
        
        loss = 0
        count = 0
        batch_size = embeddings.size(0)

        # 4. Triplet Mining dựa trên độ tương đồng nhãn
        # Với mỗi anchor i, ta tìm positive p và negative n
        for i in range(batch_size):
            # Positive là những ảnh có Jaccard similarity > 0
            # Negative là những ảnh có Jaccard similarity thấp hơn hoặc bằng 0
            
            # Để đơn giản và hiệu quả, ta lấy:
            # Anchor-Positive: Cặp có Jaccard cao nhất (khác chính nó)
            # Anchor-Negative: Cặp có Jaccard thấp nhất
            
            # Lấy các index có độ tương đồng lớn hơn 0 (trừ chính nó)
            pos_mask = (jaccard_matrix[i] > 0)
            pos_mask[i] = False # Loại bỏ chính mình
            
            neg_mask = (jaccard_matrix[i] == 0)

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Tính trọng số: Càng giống nhau về nhãn, weight càng cao
            # Giúp kéo các ca bệnh cực kỳ giống nhau lại gần nhau mạnh mẽ hơn
            weight_p = jaccard_matrix[i][pos_mask]
            
            d_p = dist_matrix[i][pos_mask]
            d_n = dist_matrix[i][neg_mask]

            # Hard Negative Mining: Lấy d_n nhỏ nhất (ca dễ nhầm nhất)
            hard_d_n = d_n.min()
            
            # Tính loss có trọng số
            # d_p - hard_d_n + margin
            current_loss = F.relu(d_p - hard_d_n + self.margin)
            loss += (current_loss * weight_p).mean()
            count += 1

        return loss / (count + 1e-8), torch.tensor(0.0) 


# ============================================================================
# ConceptCLIP Losses: IT-Align + RC-Align (from ConceptCLIP paper, arXiv:2501.15579)
# ============================================================================

class ITAlignLoss(nn.Module):
    """Image-Text Alignment Loss (SigLIP-style sigmoid contrastive).
    
    From the paper Section 3.2:
    L_IT = -1/|B| * Σ_m Σ_n log( 1 / (1 + exp(z_mn * (-t * x_m·y_n + b))) )
    
    where z_mn = +1 if (m,n) is a matching pair, -1 otherwise,
    t = logit_scale (learnable temperature), b = logit_bias (learnable).
    """
    
    def __init__(self):
        super(ITAlignLoss, self).__init__()
    
    def forward(self, image_features, text_features, logit_scale, logit_bias=None):
        """
        Args:
            image_features: (B, D) L2-normalized image CLS embeddings
            text_features: (B, D) L2-normalized text CLS embeddings
            logit_scale: scalar or (1,) learnable temperature (log scale)
            logit_bias: scalar or (1,) learnable bias (optional, default 0)
        
        Returns:
            loss: scalar IT-Align loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        B = image_features.size(0)
        
        # Compute similarity matrix: (B, B)
        # logit_scale: clamp to prevent overflow. Typical range: exp(-5) to exp(5) = 0.007 to 148
        # Original CLIP uses ~exp(2.66) ≈ 14.3
        if logit_scale.dim() == 0 or logit_scale.numel() == 1:
            # Use clamp with requires_grad preserved - clamp doesn't stop gradients within bounds
            # Check if value is reasonable before clamping
            raw_scale = logit_scale.item()
            if raw_scale < -10 or raw_scale > 10:
                print(f"[WARNING] logit_scale={raw_scale:.4f} outside safe range, clamping")
            
            # Don't clamp if within reasonable range to preserve gradients better
            if -8 < raw_scale < 8:
                t = logit_scale.exp()
            else:
                clamped_log_scale = torch.clamp(logit_scale, -10, 10)
                t = clamped_log_scale.exp()
        else:
            # Already a scale, clamp directly
            t = torch.clamp(logit_scale, 0.1, 100.0)
        
        # Compute cosine similarity
        cos_sim = image_features @ text_features.T  # (B, B) in [-1, 1]
        logits = t * cos_sim
        
        if logit_bias is not None:
            logits = logits + logit_bias
        
        # Clamp logits to prevent overflow in logsigmoid
        logits = torch.clamp(logits, -50, 50)
        
        # z_mn: +1 for matching pairs (diagonal), -1 for non-matching
        z = 2 * torch.eye(B, device=logits.device) - 1  # +1 on diag, -1 off-diag
        
        # log(sigmoid(z * logits)) = -log(1 + exp(-z * logits))
        loss = -F.logsigmoid(z * logits).mean()  # mean instead of sum then divide
        
        # Check for inf/nan
        if not torch.isfinite(loss):
            print(f"WARNING: IT-Align loss is {loss.item()}, logit_scale={logit_scale.item():.3f}, t={t.item():.3f}")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return loss


class RCAlignLoss(nn.Module):
    """Region-Concept Alignment Loss.
    
    From the paper Section 3.3:
    - Build concept embeddings by encoding each concept name individually and 
      using mean_pooling on the concept token embeddings.
    - Compute similarity matrix A_ij = cos(patch_i, concept_j)
    - S(I, T) = (1/w) * Σ_j max_i(A_ij)  -- max over patches for each concept
    - L_RC = -1/|B| * Σ_m Σ_n z_mn * S(I_m, T_n)
    
    Uses sigmoid formulation consistent with IT-Align.
    """
    
    def __init__(self):
        super(RCAlignLoss, self).__init__()
    
    def forward(self, image_token_features, concept_text_features_list, 
                logit_scale, logit_bias=None):
        """
        Args:
            image_token_features: (B, N_patches, D) patch-level image embeddings
            concept_text_features_list: list of B elements, each is (w_i, D) tensor 
                with concept embeddings for that sample (w_i = number of active concepts).
                Can have variable lengths; samples with 0 concepts are skipped.
            logit_scale: learnable temperature
            logit_bias: learnable bias (optional)
        
        Returns:
            loss: scalar RC-Align loss
        """
        B = image_token_features.size(0)
        device = image_token_features.device
        
        # Filter samples that have at least 1 concept
        valid_indices = [i for i, c in enumerate(concept_text_features_list) 
                         if c is not None and c.size(0) > 0]
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        V = len(valid_indices)
        
        # Compute temperature with numerical stability
        if logit_scale.dim() == 0 or logit_scale.numel() == 1:
            clamped_log_scale = torch.clamp(logit_scale, -10, 10)
            t = clamped_log_scale.exp()
        else:
            t = torch.clamp(logit_scale, 0.1, 100.0)
        
        # Compute S(I_m, T_n) for all valid pairs
        # S(I, T) = (1/w) * Σ_j max_i( cos(patch_i, concept_j) )
        similarity_scores = []
        
        for m_idx in valid_indices:
            img_patches = F.normalize(image_token_features[m_idx], dim=-1)  # (N, D)
            row_scores = []
            
            for n_idx in valid_indices:
                concept_embeds = concept_text_features_list[n_idx]  # (w, D)
                concept_embeds = F.normalize(concept_embeds, dim=-1)
                w = concept_embeds.size(0)
                
                # A_ij = cos(patch_i, concept_j): (N, w)
                A = img_patches @ concept_embeds.T
                
                # S(I, T) = (1/w) * Σ_j max_i(A_ij)
                max_per_concept = A.max(dim=0).values  # (w,)
                S = max_per_concept.mean()  # scalar
                
                row_scores.append(S)
            
            similarity_scores.append(torch.stack(row_scores))
        
        # similarity_scores: (V, V) matrix
        sim_matrix = torch.stack(similarity_scores)  # (V, V)
        
        # Apply temperature and bias
        logits = t * sim_matrix
        if logit_bias is not None:
            logits = logits + logit_bias
        
        # Clamp logits
        logits = torch.clamp(logits, -50, 50)
        
        # SigLIP-style loss on region-concept similarities
        z = 2 * torch.eye(V, device=device) - 1
        loss = -F.logsigmoid(z * logits).mean()
        
        # Check for inf/nan
        if not torch.isfinite(loss):
            print(f"WARNING: RC-Align loss is inf/nan, returning 0")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class ConceptCLIPLoss(nn.Module):
    """Combined ConceptCLIP loss: IT-Align + α * RC-Align.
    
    Paper uses α = 0.5 (Table 2, ablation study).
    """
    
    def __init__(self, alpha=0.5):
        super(ConceptCLIPLoss, self).__init__()
        self.alpha = alpha
        self.it_align = ITAlignLoss()
        self.rc_align = RCAlignLoss()
    
    def forward(self, image_features, text_features, image_token_features,
                concept_text_features_list, logit_scale, logit_bias=None):
        """
        Args:
            image_features: (B, D) global image CLS embeddings
            text_features: (B, D) global text CLS embeddings
            image_token_features: (B, N_patches, D) patch-level image embeddings
            concept_text_features_list: list of B tensors, each (w_i, D)
            logit_scale: learnable temperature
            logit_bias: learnable bias
        
        Returns:
            total_loss: scalar combined loss
            it_loss: scalar IT-Align loss (for logging)
            rc_loss: scalar RC-Align loss (for logging)
        """
        it_loss = self.it_align(image_features, text_features, logit_scale, logit_bias)
        rc_loss = self.rc_align(image_token_features, concept_text_features_list,
                                logit_scale, logit_bias)
        
        total_loss = it_loss + self.alpha * rc_loss
        
        return total_loss, it_loss, rc_loss