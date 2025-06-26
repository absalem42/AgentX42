# Identity Employees in Surveillance CCTV - Kaggle Solution

This repository contains a Jupyter notebook with the full training and inference pipeline used for the Kaggle competition **[Identity Employees in Surveillance CCTV](https://www.kaggle.com/competitions/identity-employees-in-surveillance-cctv)**. The notebook builds an employee face recognition model with PyTorch and an embedding-based gallery approach.

## Repository structure

- **`AgentX42.ipynb`** – Kaggle notebook implementing dataset preparation, model training, gallery creation and final submission.

## Approach

1. **Dataset setup and sanity checks.** Paths to the competition data are defined, and missing image entries in `labels.csv` are removed.
2. **Frame extraction.** For each employee video, 500 frames are sampled to augment the reference photos.
3. **Dataset & transforms.** The dataset mixes CCTV images and reference photos. A stratified split keeps 10% of CCTV images for validation. Extensive data augmentation is applied to the training set.
4. **EfficientNet‑B0 + ArcFace training.** An EfficientNet‑B0 backbone with an ArcFace head is fine‑tuned for 70 epochs using cross‑entropy loss and label smoothing.
5. **FIQA‑filtered gallery & threshold search.** A gallery of embeddings is created from high‑quality reference images and optionally video frames. FIQA is used to drop the lowest quality 30% of faces, and a grid search finds the best cosine‑similarity threshold.
6. **Hybrid soft‑max and gallery inference.** During prediction the model combines soft‑max scores with gallery cosine similarity to classify known employees or label them as `unknown`.

## Usage

The notebook is designed to run on Kaggle. Upload `AgentX42.ipynb` to a Kaggle notebook, attach the competition dataset, and execute all cells sequentially. A `submission.csv` file will be written to `/kaggle/working/` when inference finishes.

## Acknowledgements

The solution uses standard PyTorch libraries, `timm`, `torchmetrics`, and `insightface` for FIQA filtering. See the notebook for full details.
