# Signature Forgery Detection

Three pair-wise verifiers (reference genuine signature vs questioned signature):

1. Preprocessing + contour analysis + geometric features
2. HOG features + SVM (on the HOG difference of the pair)
3. Siamese CNN

Dataset: CEDAR (55 writers, 24 genuine + 24 forged each), from
https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets — included under `CEDAR/CEDAR/`.
Check the original CEDAR licence terms before reuse.

## Run
```
pip install opencv-python numpy scikit-learn torch matplotlib
python signature_verify.py --data CEDAR/CEDAR --epochs 10 --plots plots
python signature_verify.py --demo --plots plots_demo    # synthetic data, no dataset needed
```
Writers are split 60/15/25 (train/validation/test, no writer in two groups). The accept
threshold is tuned on validation writers (FAR ≈ FRR) and scored on unseen test writers.

## Results (CEDAR, 14 unseen test writers, see results_cedar.log)
| Method | Accuracy | FAR | FRR |
|---|---|---|---|
| Geometric | 0.610 | 0.360 | 0.420 |
| HOG + SVM | 0.708 | 0.225 | 0.360 |
| Siamese CNN | 0.611 | 0.393 | 0.385 |
