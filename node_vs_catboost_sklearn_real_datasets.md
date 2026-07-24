# NODE vs CatBoost on Real sklearn Datasets (Repeated Holdout)

## Setup
- Datasets: breast_cancer, wine, digits, iris (from sklearn).
- Validation: StratifiedShuffleSplit with 8 repeated holdout splits (test_size=0.2, random_state=42).
- NODE: NODEClassifier defaults on GPU (device="cuda").
- CatBoost: CatBoostClassifier defaults (standard settings, no tuning).
- Reported metrics: mean ± std across repeated holdouts.

## Dataset Info

| Dataset | Samples | Features | Classes |
|:--|--:|--:|--:|
| breast_cancer | 569 | 30 | 2 |
| wine | 178 | 13 | 3 |
| digits | 1797 | 64 | 10 |
| iris | 150 | 4 | 3 |

## Repeated Holdout Results (mean ± std)

| Dataset | Model | Device | Accuracy | ROC-AUC | LogLoss | Fit Time (s) |
|:--|:--|:--|--:|--:|--:|--:|
| breast_cancer | NODEClassifier (defaults) | cuda:0 | 0.9441 ± 0.0187 | 0.9857 ± 0.0098 | 0.1543 ± 0.0651 | 12.65 ± 0.57 |
| breast_cancer | CatBoostClassifier (defaults) | cpu(default) | 0.9671 ± 0.0154 | 0.9936 ± 0.0077 | 0.0864 ± 0.0502 | 3.09 ± 0.08 |
| wine | NODEClassifier (defaults) | cuda:0 | 0.8646 ± 0.0502 | 0.9720 ± 0.0114 | 0.4379 ± 0.0393 | 5.36 ± 0.02 |
| wine | CatBoostClassifier (defaults) | cpu(default) | 0.9757 ± 0.0232 | 0.9991 ± 0.0017 | 0.0844 ± 0.0580 | 1.33 ± 0.09 |
| digits | NODEClassifier (defaults) | cuda:0 | 0.9747 ± 0.0050 | 0.9995 ± 0.0002 | 0.1082 ± 0.0139 | 37.16 ± 0.09 |
| digits | CatBoostClassifier (defaults) | cpu(default) | 0.9806 ± 0.0065 | 0.9997 ± 0.0002 | 0.0676 ± 0.0146 | 5.34 ± 0.05 |
| iris | NODEClassifier (defaults) | cuda:0 | 0.9583 ± 0.0295 | 0.9958 ± 0.0068 | 0.2246 ± 0.0462 | 4.51 ± 0.03 |
| iris | CatBoostClassifier (defaults) | cpu(default) | 0.9542 ± 0.0248 | 0.9935 ± 0.0076 | 0.1744 ± 0.0921 | 0.38 ± 0.03 |

## Winners by Metric (higher is better except LogLoss)

| Dataset | Best Accuracy | Best ROC-AUC | Best LogLoss |
|:--|:--|:--|:--|
| breast_cancer | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) |
| digits | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) |
| iris | NODEClassifier (defaults) | NODEClassifier (defaults) | CatBoostClassifier (defaults) |
| wine | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) | CatBoostClassifier (defaults) |

## Notes
- CatBoost was run with standard/default settings as requested.
- NODE was run with default settings except explicit GPU device selection.
- Small datasets (especially wine and iris) still show variance despite repeats.
