# Master Prompt for Hybrid Pilot/Production Multimodal Project (Updated)

You are an expert AI agent specializing in PyTorch, Hugging Face Transformers, and Parameter-Efficient Fine-Tuning (PEFT). 
Your task is to write a single, complete, production-ready Jupyter Notebook pipeline for a university assignment on Large Multimodal Models (LMMs).
The target is to perform Zero-Shot and Fine-Tuning classification on the MedMNIST (DermaMNIST) dataset using a Light-Weight LMM (CLIP: `openai/clip-vit-base-patch32`).

---

### CRITICAL INFRASTRUCTURE RULE (DO NOT VIOLATE):
- DO NOT embed any `pip install`, `subprocess.check_call`, or automatic library installations inside the main notebook cells. 
- Assume all requirements (`transformers`, `medmnist`, `scikit-learn`, `pandas`, `pillow`, `matplotlib`) are already natively installed in the environment.
- Assume the Hugging Face weights for `openai/clip-vit-base-patch32` are already fully cached on the local disk.

---

### 1. DYNAMIC ENVIRONMENT & RUN-TIME CONFIGURATION (The Switch)
At the very top of the script, implement a clean configuration system that detects the infrastructure by checking if `os.path.exists('/kaggle/working')` is True.

- **IF DETECTED AS LOCAL (Pilot Mode):**
  * Objective: Absolute minimum computational load. Output quality does not matter; code integrity matters.
  * DEVICE = "cpu" (Force CPU to prevent local CUDA overhead).
  * BATCH_SIZE = 1 (Zero RAM stress).
  * IMAGE_SIZE = 224 (Mandatory for CLIP input size, but applied ONLY to the debug subset).
  * EPOCHS = 1
  * DEBUG_SUBSET = True (Slice the dataset to ONLY the first 10 training images and 5 testing images).

- **IF DETECTED AS KAGGLE (Production Mode):**
  * Objective: Fully optimized, rigorous training on high-performance accelerators.
  * DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  * BATCH_SIZE = 32
  * IMAGE_SIZE = 224
  * EPOCHS = 5
  * DEBUG_SUBSET = False (Train and evaluate on the 100% full dataset).

---

### 2. DATASET INGESTION & PIPELINE (MedMNIST API)
- Programmatically load `DermaMNIST` using the `medmnist` Python package.
- If `DEBUG_SUBSET` is True, intercept the PyTorch Dataset instantiation and forcefully truncate the indices (`dataset.imgs`, `dataset.labels`) to match the local micro-slice configuration.
- Extract the classification textual labels dynamically from `info['label']`.

### 3. MULTIMODAL ZERO-SHOT CLASSIFICATION
- Load `openai/clip-vit-base-patch32` and its standard processor.
- Formulate text prompts dynamically using the class names (e.g., `"A clinical image of {class_name}"`).
- Extract normalized image embeddings and text embeddings. Compute cosine similarity scores, apply softmax over logits to derive class predictions, and evaluate the Zero-Shot baseline.

### 4. LINEAR PROBING & PARAMETER-EFFICIENT ADAPTATION
- Implement a custom PyTorch module (`CLIPLinearProber`). 
- Freeze 100% of the native CLIP base weights (Vision & Text Encoders) to ensure zero gradient calculation on heavy backpropagation.
- Append a lightweight, trainable Classification Head (Linear Layers) at the output of the image encoder.
- Write a clean training loop tracking Cross-Entropy Loss over the dynamically determined `EPOCHS`.

### 5. SCHOLARLY METRICS & ANALYSIS COMPLIANCE
- Integrate `scikit-learn` metrics to calculate and log the macro Area Under the ROC Curve (AUC) and global Accuracy (ACC) for both Zero-Shot and the trained linear probing pipeline.
- Include a structural safeguard to handle multi-class AUC edge cases if the local micro-subset lacks representations of all classes.

### 6. EVALUATION INFERENCE WORKFLOW (Teacher Verification Ready)
Implement a robust, standalone utility function named `predict_for_evaluation(image_folder_path, output_csv_path)`.
- It must verify the folder exists, read all standard images sequentially, and process them via the configuration's designated `DEVICE`.
- It must strip filename extensions (extracting `img_001` from `/path/img_001.jpg`).
- Run the inference pipeline and output a CSV file containing exactly two columns: `image_id` (the name string) and `class_id` (the predicted integer index).