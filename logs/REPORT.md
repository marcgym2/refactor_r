# Model Comparison Report

## Winner
Winner: **tabicl_v2** with macro F1 `0.288576` and accuracy `0.301667`.
It beat `tabpfn_v2` by `0.046326` macro F1 on the held-out test split.

## Why This Protocol
The full feature table contains four shift variants of 28-day windows, and those windows overlap heavily in calendar time.
To avoid leakage, the comparison was restricted to `Shift == 0`, which matches the live inference path and produces non-overlapping intervals.
The baseline is the current live FFNN architecture retrained with the repo's existing loss and hyperparameters, using a middle dev block for early stopping and the final block as the only scorecard.

## Data Audit Highlights
- Raw feature table shape: `22100 x 79`
- Standardized feature table shape: `22100 x 79`
- Comparison frame shape: `5600 x 79`
- Full-frame cross-shift overlap pairs: `657`
- Shift-0 split overlap counts train/dev/test: `0` / `0` / `0`
- Comparison-frame total nulls after preprocessing: `0`

## Final Table
   model_name  accuracy  macro_f1  train_seconds  predict_seconds device  f1_class_1  f1_class_2  f1_class_3  f1_class_4  f1_class_5
    tabicl_v2  0.301667  0.288576       0.598714       117.008653    cpu    0.351648    0.253521    0.351145    0.152047    0.334520
    tabpfn_v2  0.286667  0.242250       0.427213       404.106188    cpu    0.344322    0.129412    0.395210    0.031496    0.310811
baseline_ffnn  0.296667  0.236917       0.183449         0.000250    cpu    0.432304    0.325879    0.192090    0.218182    0.016129

## Winner Reasoning
`tabicl_v2` is the best choice here because it had the strongest macro F1 on a balanced five-class target, which is the most relevant metric for quintile prediction.
Its per-class F1 profile was: class 1 = 0.3516, class 2 = 0.2535, class 3 = 0.3511, class 4 = 0.1520, class 5 = 0.3345.
Accuracy is included as a secondary metric, but macro F1 gets priority because the goal is robust rank-bucket discrimination across all quintiles.

## Caveats
- The original all-shifts table is leaky for naive split-based evaluation because shifted windows overlap in time.
- The baseline dev split is used for early stopping; challengers are zero-shot / fixed-default models and do not use that split for tuning.
