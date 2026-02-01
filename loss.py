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