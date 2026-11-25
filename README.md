# Machine Learning Assignment 3 — m26csa004

PyTorch implementation for CIFAR-10: MLPs, CNNs, convolutional autoencoders,
SVMs and clustering.

## Identity and reproducibility
- Roll number: `m26csa004`
- Seed: `26004` (all numeric characters of the roll number, in order)
- Dataset: CIFAR-10
- Split: stratified 45,000 train / 5,000 validation from the official 50,000
  training images; official 10,000-image test set held out.
- No pretrained models or external data are used.
- Validation is used for model selection; test data is evaluated after choices
  are fixed.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
Run each question separately (recommended for debugging and checkpointing):

```bash
python src/q1.py --device auto
python src/q2.py --device auto
python src/q3.py --device auto
python src/q4.py
```

Or run the sequence:

```bash
python run_all.py
```

CIFAR-10 downloads automatically into `./data`. The exact train/validation
indices are persisted as `data/split_seed_26004.npz`. Outputs, checkpoints,
CSV histories, metrics and figures are written to `./outputs`.

For a CPU-only run use `--device cpu`. For a smaller smoke test, reduce
`--epochs`, `--mem-epochs`, and `--subset-per-class` when invoking scripts.
Smoke-test results are not final assignment results.

## Experiment protocol
### Q1
- Q1(a): compare same MLP topology with and without dropout to examine the
  regularisation/generalisation explanation. Record exact parameter counts.
- Q1(b): train a high-capacity MLP against fixed random training labels and
  evaluate against untouched true validation labels. Accuracy/loss trajectories
  show memorisation.

### Q2
- Clean MLP and CNN are trained independently.
- Q2(b): one fixed permutation of the 1,024 spatial positions is applied to
  every RGB image, with RGB triplets preserved. The same permutation is used
  for train, validation and test; both models are retrained.
- Q2(c): central 12x12 zero occlusion is applied to test images for the already
  trained clean CNN. Confidence and error summaries are saved.

### Q3
- Three convolutional autoencoders differ only in latent vector dimension
  (64, 512, 4096).
- Denoising AE receives Gaussian-noisy images and clean targets.
- A fixed CNN is used to compare classification of noisy images and
  reconstructions. A logistic-regression linear probe evaluates frozen latent
  representations.

### Q4 (bonus)
- StandardScaler and PCA are fitted on training features only.
- SVM uses a fixed stratified subset (1,000/class by default) chosen once
  before the C/kernel sweep. Same subset is used for every setting.
- C values: 0.01, 0.1, 1, 10, 100. Kernels: linear, polynomial, RBF.
- K-means is fitted on training representations. K is selected using
  validation silhouette, without validation/test labels.
- Test labels are used only for final ARI/NMI and contingency-table evaluation.

## Important notes / checks before final submission
1. Compare parameter counts before claiming a parameter-matched CNN/MLP.
   The provided starter architectures are not guaranteed to have identical
   parameter counts. Adjust widths and record the exact counts.
2. Q1(a) dropout changes regularisation but not parameter count; state this
   as the controlled variable. It tests regularisation, not every possible
   explanation for poor MLP performance.
3. The Q2(c) occlusion experiment is a first controlled probe. Report
   accuracy, mean confidence, correct/incorrect confidence and confusion
   matrix. Add per-class errors if space permits.
4. The Q3 clean-AE baseline is trained on clean inputs; denoising AE uses
   noisy inputs and clean targets. Repeat noisy evaluation with a fixed
   generator/seed for strict stochastic reproducibility if comparing runs.
5. Q4 PCA/SVM may be computationally expensive. The fixed stratified SVM
   subset is allowed by the assignment; report its exact size and indices.
6. Run all experiments on the same hardware/software environment and retain
   logs. Do not fill the report with invented results.

## Outputs
- `outputs/q1`: MLP histories, checkpoints, memorisation CSV/plot.
- `outputs/q2`: clean/permuted checkpoints, spatial permutation, summary,
  occlusion metrics.
- `outputs/q3`: autoencoder checkpoints, histories and metrics.
- `outputs/q4`: SVM sweep, selected model metrics, difficult pairs,
  clustering metrics and contingency table.

## Additional analysis artifacts
- Q2 starter model parameter counts are closely matched (about 1.71M MLP
  versus 1.75M CNN); exact values are printed at runtime.
- Q4 writes `difficult_pairs_boundary.json` and one NPZ per selected class
  pair containing training support-vector indices and test examples closest
  to the binary decision boundary. Use those indices to render image grids.
- `report_cvpr.tex` is a CVPR two-column report scaffold. Fill all result
  placeholders only after running the experiments.

## CVPR report compilation
`report_cvpr.tex` is the two-column report source. It expects the official
CVPR LaTeX style file (`cvpr.sty`) and associated template files from the
conference template package in the same working directory. The PDF is
intentionally not fabricated: fill in measured result placeholders after
running the code, then compile using the official style package.
