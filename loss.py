"""
Pytorch adaptation of https://omoindrot.github.io/triplet-loss
https://github.com/omoindrot/tensorflow-triplet-loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, mining="batch_all"):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.mining = mining

        if mining == "batch_all":
            self.loss_fn = batch_all_triplet_loss
        if mining == "batch_hard":
            self.loss_fn = batch_hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        logits = torch.matmul(embeddings, embeddings.t()) / self.temperature

        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        if labels.dim() == 1:
            positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        else:
            intersection = torch.matmul(labels.float(), labels.float().t())
            positive_mask = intersection > 0

        positive_mask = positive_mask & ~self_mask
        logits = logits.masked_fill(self_mask, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor = positives_per_anchor > 0
        if not valid_anchor.any():
            return embeddings.sum() * 0.0

        loss = -(positive_mask.float() * log_prob).sum(dim=1)
        loss = loss[valid_anchor] / (positives_per_anchor[valid_anchor].float() + self.eps)
        return loss.mean()


def batch_hard_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # hardest positive for every anchor
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # Add max value in each row to invalid negatives
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
        1.0 - mask_anchor_negative
    )

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

    fraction_positive_triplets = num_positive_triplets / (
        num_valid_triplets.float() + 1e-16
    )

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
            pos_mask = jaccard_matrix[i] > 0
            pos_mask[i] = False  # Loại bỏ chính mình

            neg_mask = jaccard_matrix[i] == 0

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

        if count == 0:
            return embeddings.sum() * 0.0, embeddings.new_tensor(0.0)

        return loss / count, embeddings.new_tensor(0.0)


class JaccardSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def compute_jaccard_sim(self, labels):
        # labels: [B, C] multi-hot
        intersection = torch.matmul(labels, labels.t())  # [B, B]
        label_sums = labels.sum(dim=1, keepdim=True)
        union = label_sums + label_sums.t() - intersection
        return intersection / (union + self.eps)

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D]
        labels: [B, C] (multi-hot)
        """

        B = embeddings.size(0)

        # Normalize embeddings → cosine similarity
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())  # [B, B]
        sim_matrix = sim_matrix / self.temperature

        # Jaccard similarity as weights
        jaccard = self.compute_jaccard_sim(labels)  # [B, B]

        # Mask out self-comparisons
        self_mask = torch.eye(B, device=embeddings.device).bool()
        jaccard = jaccard.masked_fill(self_mask, 0.0)

        # Normalize weights per anchor (important!)
        weight_sum = jaccard.sum(dim=1, keepdim=True) + self.eps
        weights = jaccard / weight_sum  # [B, B]

        # Log-softmax over rows
        log_prob = sim_matrix - torch.logsumexp(
            sim_matrix.masked_fill(self_mask, -1e9), dim=1, keepdim=True
        )

        # Final loss
        loss = -(weights * log_prob).sum(dim=1)

        # Only keep anchors that have at least one positive
        valid_mask = weight_sum.squeeze() > self.eps
        loss = loss[valid_mask]

        if loss.numel() == 0:
            return embeddings.new_tensor(0.0)

        return loss.mean()


def compute_multilabel_masks_and_weights(
    labels: torch.Tensor,
    use_jaccard_weight: bool = True,
    eps: float = 1e-8,
):
    labels = labels.float()
    intersection = labels @ labels.t()
    label_cardinality = labels.sum(dim=1, keepdim=True)
    union = label_cardinality + label_cardinality.t() - intersection
    jaccard = intersection / union.clamp_min(eps)

    batch_size = labels.size(0)
    eye_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
    positive_mask = (intersection > 0) & ~eye_mask
    negative_mask = (intersection == 0) & ~eye_mask

    if use_jaccard_weight:
        positive_weights = jaccard * positive_mask.float()
    else:
        positive_weights = positive_mask.float()

    return positive_mask, negative_mask, positive_weights


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos

        if self.clip is not None and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(prob_pos.clamp_min(self.eps))
        log_neg = torch.log(prob_neg.clamp_min(self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
            loss = loss * focal_weight

        return -loss.sum(dim=1).mean()


class MultiLabelContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        use_jaccard_weight: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_jaccard_weight = use_jaccard_weight
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        _, _, positive_weights = compute_multilabel_masks_and_weights(
            labels=labels,
            use_jaccard_weight=self.use_jaccard_weight,
            eps=self.eps,
        )

        logits = embeddings @ embeddings.t()
        logits = logits / self.temperature

        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        positive_weight_sums = positive_weights.sum(dim=1)
        valid_anchors = positive_weight_sums > 0
        if not valid_anchors.any():
            return embeddings.sum() * 0.0

        weighted_log_prob = (positive_weights * log_prob).sum(dim=1)
        loss = -weighted_log_prob[valid_anchors] / positive_weight_sums[
            valid_anchors
        ].clamp_min(self.eps)
        return loss.mean()


class DualBranchMultiLabelLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 0.07,
        use_jaccard_weight: bool = True,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.contrastive = MultiLabelContrastiveLoss(
            temperature=temperature,
            use_jaccard_weight=use_jaccard_weight,
        )
        self.asl = AsymmetricLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )

    def forward(self, outputs, labels):
        if not isinstance(outputs, dict):
            raise TypeError(
                "DualBranchMultiLabelLoss expects model output with "
                "'embedding' and 'logits' keys."
            )
        if "embedding" not in outputs or "logits" not in outputs:
            raise KeyError(
                "DualBranchMultiLabelLoss expects model output with "
                "'embedding' and 'logits' keys."
            )

        contrastive_loss = self.contrastive(outputs["embedding"], labels)
        asl_loss = self.asl(outputs["logits"], labels)
        total_loss = contrastive_loss + self.alpha * asl_loss
        return total_loss, {
            "contrastive": contrastive_loss.detach(),
            "asl": asl_loss.detach(),
        }


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
            logit_scale: scalar learnable temperature in LOG space (typical: 2.6-4.6)
            logit_bias: scalar learnable bias (optional, default 0)

        Returns:
            loss: scalar IT-Align loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        B = image_features.size(0)

        # logit_scale is in log space: clamp to [0, ln(100)=4.6052] then exp()
        # This gives effective temperature in [1, 100], following CLIP convention
        clamped_log_scale = torch.clamp(logit_scale, min=0.0, max=4.6052)
        t = clamped_log_scale.exp()  # effective temperature in [1, 100]

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
        loss = -F.logsigmoid(z * logits).mean()

        # Check for inf/nan
        if not torch.isfinite(loss):
            print(
                f"WARNING: IT-Align loss is {loss.item()}, logit_scale={logit_scale.item():.3f}, t={t.item():.3f}"
            )
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

    def forward(
        self,
        image_token_features,
        concept_text_features_list,
        logit_scale,
        logit_bias=None,
    ):
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
        valid_indices = [
            i
            for i, c in enumerate(concept_text_features_list)
            if c is not None and c.size(0) > 0
        ]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        V = len(valid_indices)

        # logit_scale is in log space: clamp to [0, ln(100)=4.6052] then exp()
        clamped_log_scale = torch.clamp(logit_scale, min=0.0, max=4.6052)
        t = clamped_log_scale.exp()  # effective temperature in [1, 100]

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

    def forward(
        self,
        image_features,
        text_features,
        image_token_features,
        concept_text_features_list,
        logit_scale,
        logit_bias=None,
    ):
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
        rc_loss = self.rc_align(
            image_token_features, concept_text_features_list, logit_scale, logit_bias
        )

        total_loss = it_loss + self.alpha * rc_loss

        return total_loss, it_loss, rc_loss
