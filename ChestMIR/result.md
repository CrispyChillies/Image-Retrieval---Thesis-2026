=== Stage 1 - Global Retrieval ===
R@1: 91.30%, R@5: 96.66%, R@10: 97.99%
mAP: 91.87%
P@1: 91.30%, P@5: 92.11%, P@10: 92.24%
Top-1: Acc 91.30% | P_macro 91.41% | R_macro 91.30% | F1_macro 91.32%
Top-5: Acc 95.99% | P_macro 96.01% | R_macro 95.99% | F1_macro 95.97%

=== Stage 2 - Adaptive Lesion Rerank ===
R@1: 90.64%, R@5: 96.99%, R@10: 97.99%
mAP: 91.85%
P@1: 90.64%, P@5: 92.17%, P@10: 92.24%
Top-1: Acc 90.64% | P_macro 90.89% | R_macro 90.63% | F1_macro 90.70%
Top-5: Acc 96.32% | P_macro 96.33% | R_macro 96.33% | F1_macro 96.31%
Fallback(global-only): 275/299 | Reranked: 24/299 | topK=100
Adaptive lesion usage: consolidation:9, infiltration:1, lung opacity:6, pleural effusion:8

=== Stage 2 - Lesion Rerank (Consolidation) ===
R@1: 90.64%, R@5: 96.66%, R@10: 97.99%
mAP: 91.84%
P@1: 90.64%, P@5: 91.97%, P@10: 92.17%
Top-1: Acc 90.64% | P_macro 90.81% | R_macro 90.63% | F1_macro 90.69%
Top-5: Acc 95.99% | P_macro 96.01% | R_macro 95.99% | F1_macro 95.97%
Fallback(global-only): 290/299 | Reranked: 9/299 | topK=100

=== Stage 2 - Lesion Rerank (Lung Opacity) ===
R@1: 91.30%, R@5: 96.66%, R@10: 97.99%
mAP: 91.87%
P@1: 91.30%, P@5: 92.11%, P@10: 92.24%
Top-1: Acc 91.30% | P_macro 91.41% | R_macro 91.30% | F1_macro 91.32%
Top-5: Acc 95.99% | P_macro 96.01% | R_macro 95.99% | F1_macro 95.97%
Fallback(global-only): 293/299 | Reranked: 6/299 | topK=100

=== Stage 2 - Lesion Rerank (Infiltration) ===
R@1: 91.30%, R@5: 96.66%, R@10: 97.99%
mAP: 91.87%
P@1: 91.30%, P@5: 92.11%, P@10: 92.24%
Top-1: Acc 91.30% | P_macro 91.41% | R_macro 91.30% | F1_macro 91.32%
Top-5: Acc 95.99% | P_macro 96.01% | R_macro 95.99% | F1_macro 95.97%
Fallback(global-only): 298/299 | Reranked: 1/299 | topK=100

=== Stage 2 - Lesion Rerank (Atelectasis) ===
R@1: 91.30%, R@5: 96.66%, R@10: 97.99%
mAP: 91.87%
P@1: 91.30%, P@5: 92.11%, P@10: 92.24%
Top-1: Acc 91.30% | P_macro 91.41% | R_macro 91.30% | F1_macro 91.32%
Top-5: Acc 95.99% | P_macro 96.01% | R_macro 95.99% | F1_macro 95.97%
Fallback(global-only): 299/299 | Reranked: 0/299 | topK=100

=== Stage 2 - Lesion Rerank (Pleural effusion) ===
R@1: 91.30%, R@5: 96.99%, R@10: 97.99%
mAP: 91.87%
P@1: 91.30%, P@5: 92.31%, P@10: 92.31%
Top-1: Acc 91.30% | P_macro 91.47% | R_macro 91.30% | F1_macro 91.34%
Top-5: Acc 96.32% | P_macro 96.33% | R_macro 96.33% | F1_macro 96.31%
Fallback(global-only): 290/299 | Reranked: 9/299 | topK=100

=== Final Summary Across 5 Lesions ===
Mean mAP: 91.86%
Mean R@1: 91.17%
Mean R@5: 96.72%
Per-lesion:
- Consolidation: mAP 91.84% | R@1 90.64% | fallback 290 | reranked 9
- Lung Opacity: mAP 91.87% | R@1 91.30% | fallback 293 | reranked 6
- Infiltration: mAP 91.87% | R@1 91.30% | fallback 298 | reranked 1
- Atelectasis: mAP 91.87% | R@1 91.30% | fallback 299 | reranked 0
- Pleural effusion: mAP 91.87% | R@1 91.30% | fallback 290 | reranked 9