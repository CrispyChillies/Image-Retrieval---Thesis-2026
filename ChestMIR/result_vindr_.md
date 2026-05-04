=== Stage 1 - Global Retrieval ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%

=== Stage 2 - Adaptive Lesion Rerank ===
R@1: 59.37%, R@5: 70.90%, R@10: 72.30%
mAP: 51.56%
P@1: 59.37%, P@5: 57.65%, P@10: 57.16%
Top-1: Acc 59.37% | P_macro 0.40% | R_macro 0.41% | F1_macro 0.39%
Top-5: Acc 67.97% | P_macro 0.40% | R_macro 0.35% | F1_macro 0.32%
Fallback(global-only): 2662/3000 | Reranked: 338/3000 | Candidate-match queries: 338 | Candidate match rate in topK: 0.49% | topK=200
Adaptive lesion usage: calcification:15, cardiomegaly:171, consolidation:2, ild:22, lung opacity:1, nodule mass:46, pleural effusion:42, pleural thickening:7, pulmonary fibrosis:32

=== Stage 2 - Lesion Rerank (Consolidation) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2998/3000 | Reranked: 2/3000 | Candidate-match queries: 2 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Lung Opacity) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2999/3000 | Reranked: 1/3000 | Candidate-match queries: 1 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Infiltration) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Atelectasis) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Pleural effusion) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.51% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2955/3000 | Reranked: 45/3000 | Candidate-match queries: 45 | Candidate match rate in topK: 0.02% | topK=200

=== Stage 2 - Lesion Rerank (Nodule/Mass) ===
R@1: 59.23%, R@5: 69.70%, R@10: 71.00%
mAP: 51.54%
P@1: 59.23%, P@5: 57.35%, P@10: 56.88%
Top-1: Acc 59.23% | P_macro 0.53% | R_macro 0.44% | F1_macro 0.43%
Top-5: Acc 67.57% | P_macro 0.23% | R_macro 0.27% | F1_macro 0.24%
Fallback(global-only): 2951/3000 | Reranked: 49/3000 | Candidate-match queries: 49 | Candidate match rate in topK: 0.03% | topK=200

=== Stage 2 - Lesion Rerank (Cardiomegaly) ===
R@1: 59.17%, R@5: 70.57%, R@10: 71.93%
mAP: 51.55%
P@1: 59.17%, P@5: 57.59%, P@10: 57.12%
Top-1: Acc 59.17% | P_macro 0.34% | R_macro 0.33% | F1_macro 0.33%
Top-5: Acc 67.87% | P_macro 0.25% | R_macro 0.31% | F1_macro 0.28%
Fallback(global-only): 2815/3000 | Reranked: 185/3000 | Candidate-match queries: 185 | Candidate match rate in topK: 0.44% | topK=200

=== Stage 2 - Lesion Rerank (Edema) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Pneumothorax) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Pleural thickening) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2990/3000 | Reranked: 10/3000 | Candidate-match queries: 10 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Pulmonary fibrosis) ===
R@1: 59.07%, R@5: 69.53%, R@10: 70.83%
mAP: 51.52%
P@1: 59.07%, P@5: 57.32%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2957/3000 | Reranked: 43/3000 | Candidate-match queries: 43 | Candidate match rate in topK: 0.03% | topK=200

=== Stage 2 - Lesion Rerank (Enlarged PA) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (ILD) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 2967/3000 | Reranked: 33/3000 | Candidate-match queries: 33 | Candidate match rate in topK: 0.02% | topK=200

=== Stage 2 - Lesion Rerank (Calcification) ===
R@1: 59.10%, R@5: 69.50%, R@10: 70.80%
mAP: 51.52%
P@1: 59.10%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.10% | P_macro 0.49% | R_macro 0.38% | F1_macro 0.41%
Top-5: Acc 67.53% | P_macro 0.34% | R_macro 0.26% | F1_macro 0.24%
Fallback(global-only): 2970/3000 | Reranked: 30/3000 | Candidate-match queries: 30 | Candidate match rate in topK: 0.02% | topK=200

=== Stage 2 - Lesion Rerank (Lung cavity) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Stage 2 - Lesion Rerank (Lung cyst) ===
R@1: 59.07%, R@5: 69.47%, R@10: 70.77%
mAP: 51.52%
P@1: 59.07%, P@5: 57.31%, P@10: 56.86%
Top-1: Acc 59.07% | P_macro 0.50% | R_macro 0.37% | F1_macro 0.40%
Top-5: Acc 67.50% | P_macro 0.21% | R_macro 0.25% | F1_macro 0.22%
Fallback(global-only): 3000/3000 | Reranked: 0/3000 | Candidate-match queries: 0 | Candidate match rate in topK: 0.00% | topK=200

=== Final Summary Across 16 Lesions ===
Mean mAP: 51.53%
Mean R@1: 59.09%
Mean R@5: 69.56%
Per-lesion:
- Consolidation: mAP 51.52% | R@1 59.07% | fallback 2998 | reranked 2
- Lung Opacity: mAP 51.52% | R@1 59.07% | fallback 2999 | reranked 1
- Infiltration: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- Atelectasis: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- Pleural effusion: mAP 51.52% | R@1 59.07% | fallback 2955 | reranked 45
- Nodule/Mass: mAP 51.54% | R@1 59.23% | fallback 2951 | reranked 49
- Cardiomegaly: mAP 51.55% | R@1 59.17% | fallback 2815 | reranked 185
- Edema: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- Pneumothorax: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- Pleural thickening: mAP 51.52% | R@1 59.07% | fallback 2990 | reranked 10
- Pulmonary fibrosis: mAP 51.52% | R@1 59.07% | fallback 2957 | reranked 43
- Enlarged PA: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- ILD: mAP 51.52% | R@1 59.07% | fallback 2967 | reranked 33
- Calcification: mAP 51.52% | R@1 59.10% | fallback 2970 | reranked 30
- Lung cavity: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0
- Lung cyst: mAP 51.52% | R@1 59.07% | fallback 3000 | reranked 0