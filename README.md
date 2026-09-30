# Deep Learning-Based Fruit Quality, Ripeness and Defect Assessment

A Deep Learning PBL project. A user uploads a fruit photo and the system predicts the **fruit type**, **ripeness**, **visible defect** and the **confidence** of each prediction, then adds a simple **rule-based overall quality rating**.

> **Status note:** this repository contains code only. No model has been trained and no accuracy is claimed. All numbers must come from your own training run (see `results/` after training).

---

## 1. Problem Statement
Fruit quality is usually judged by eye. Manual inspection is slow, subjective and inconsistent between people. This project builds an image-based system that gives a fast, repeatable assessment of the *visible* condition of a fruit.

## 2. Motivation
- Reduces dependence on subjective manual sorting.
- Shows a full deep-learning workflow (data, transfer learning, evaluation, deployment) in a small, demonstrable project.
- Uses lightweight models that train on free Google Colab GPUs.

## 3. Objectives
1. Classify the fruit type (Apple, Banana, Orange, Mango).
2. Classify ripeness (Unripe, Ripe, Overripe).
3. Classify visible defect (Healthy, Bruised, Spotted, Rotten).
4. Combine the predictions into an overall quality rating using transparent, editable rules.
5. Provide a simple Streamlit web app with confidence scores.
6. Evaluate each model honestly on a held-out test set.

## 4. Features
- Three independent MobileNetV2 transfer-learning models.
- Data augmentation (flip, rotation, zoom, brightness, contrast).
- Two-phase training (train the new head, then optional fine-tuning) with early stopping and checkpointing.
- Automatic evaluation: accuracy, precision, recall, F1, confusion matrix, training curves.
- Dataset checker: folder structure, corrupt images and duplicate images across splits (data leakage).
- Rule-based quality layer that is clearly separated from the deep-learning outputs.
- Friendly error messages (no image, invalid file, unsupported format, missing models, wrong dataset layout).

## 5. Technology Stack
Python, TensorFlow/Keras (MobileNetV2, transfer learning), OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn, Streamlit, Google Colab.

## 6. System Architecture

```text
                    Uploaded image
                          |
              OpenCV: decode, BGR->RGB, resize 224x224
                          |
        +-----------------+------------------+
        |                 |                  |
   Model 1: Fruit    Model 2: Ripeness   Model 3: Defect
   (MobileNetV2)     (MobileNetV2)       (MobileNetV2)
        |                 |                  |
   fruit + conf      ripeness + conf     defect + conf
        |                 |                  |
        |                 +--------+---------+
        |                          |
        |            Rule-based quality assessment
        |            (src/quality_rules.py - NOT deep learning)
        |                          |
        +-----------> Streamlit dashboard <-------+
```

Each model:

```text
Input (224x224x3) -> Rescaling (-1..1) -> MobileNetV2 (ImageNet weights)
 -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(N, Softmax)
```

**Why three separate models?** The three questions need different visual cues (shape for fruit type, colour for ripeness, local texture/spots for defects). Separate models are easier to train, evaluate and debug, can use different datasets, and one weak model does not damage the others. The trade-off is 3x storage and inference time (still small for MobileNetV2).

## 7. Dataset Requirements

Three tasks have three different label sets, so each task has its own folder with `train / validation / test` inside it:

```text
dataset/
├── fruit/
│   ├── train/       apple/  banana/  mango/  orange/
│   ├── validation/  apple/  banana/  mango/  orange/
│   └── test/        apple/  banana/  mango/  orange/
├── ripeness/
│   ├── train/       unripe/  ripe/  overripe/
│   ├── validation/  (same three folders)
│   └── test/        (same three folders)
└── defect/
    ├── train/       healthy/  bruised/  spotted/  rotten/
    ├── validation/  (same four folders)
    └── test/        (same four folders)
```

- **The folder name is the label.** Use exactly the lowercase names above for ripeness and defect (the quality rules depend on them). The fruit folder may get extra fruits later.
- Formats: `.jpg .jpeg .png .bmp`. At least 5 images per class per split is enforced; for meaningful results aim for **hundreds per class**.
- **One image = one label per task.** The same physical photo may appear in all three tasks with its own label (e.g. a banana photo goes to `fruit/.../banana`, `ripeness/.../ripe`, `defect/.../spotted`).
- Suggested split: about 70% train, 15% validation, 15% test.

### Preventing data leakage
Data leakage = the model sees (near-)identical images in training and in validation/test, making scores look better than they really are.
1. Split **before** any augmentation. Augmentation in this project happens on the fly, only on training batches; validation and test images are never augmented.
2. Never copy pre-augmented images across splits.
3. Run `python -m src.prepare_dataset check --task <task>` - it reports exact duplicate files that appear in different splits.
4. The checker cannot detect *near*-duplicates (e.g. several photos of the same fruit from slightly different angles). If your source has such groups, keep each group in one split.
5. The test set is used only for the final evaluation, never for choosing settings.

### Public datasets (verify before use)
No single public dataset provides fruit type + ripeness + defect labels for all four fruits, so expect to **combine sources and/or photograph and label your own fruit** (a smartphone and a plain background work well). Possible starting points to search for on Kaggle / other repositories - I have not downloaded or verified their contents, so check the licence, class folders and image counts yourself:
- **Fruits-360** (Muresan & Oltean) - many fruit varieties on white backgrounds; useful for the *fruit* task.
- **"Fresh and Rotten Fruits"** style datasets (folders such as fresh/rotten apples, bananas, oranges) - useful for the *defect* task (healthy vs rotten) and partly for the *fruit* task.
- **Banana ripeness** datasets (unripe/ripe/overripe-style folders) - useful for the *ripeness* task.
- Bruised and spotted fruit images are rare; you will probably need to collect or search for them, or merge the classes.

### Dataset that already has `train/` and `test/` (fresh / rotten / unripe folders)
Keep your download OUTSIDE `dataset/`, e.g. `raw_data/train/...` and `raw_data/test/...`, then use the ready-made mapping files in `mappings/`:
```bash
python -m src.prepare_dataset split --source raw_data/train --test-source raw_data/test --task fruit    --mapping mappings/fruit.json
python -m src.prepare_dataset split --source raw_data/train --test-source raw_data/test --task ripeness --mapping mappings/ripeness.json
python -m src.prepare_dataset split --source raw_data/train --test-source raw_data/test --task defect   --mapping mappings/defect.json
```
Your `test/` folder is used unchanged as the test set; a validation set is cut from `train/`. A dataset may use only a **subset** of the class names (at least 2 per task), e.g. fruit = apple/banana/orange, ripeness = unripe/ripe, defect = healthy/rotten. The models can only predict classes they were trained on. If "fresh" images are mapped to `ripe`, state this assumption in your report.

### Adapting a downloaded dataset
Use the helper script to pool folders, remove exact duplicates, and split them randomly into train/validation/test.

Example: a download with folders `freshapples`, `rottenapples`, `freshbanana`, `rottenbanana` (your actual names will differ - list them first).

`mapping_fruit.json` (folder name -> fruit class; fresh and rotten images of the same fruit merge):
```json
{"freshapples": "apple", "rottenapples": "apple", "freshbanana": "banana", "rottenbanana": "banana"}
```
`mapping_defect.json` (folder name -> defect class):
```json
{"freshapples": "healthy", "freshbanana": "healthy", "rottenapples": "rotten", "rottenbanana": "rotten"}
```
```bash
python -m src.prepare_dataset split --source path/to/download --task fruit  --mapping mapping_fruit.json
python -m src.prepare_dataset split --source path/to/download --task defect --mapping mapping_defect.json
python -m src.prepare_dataset check --task fruit
```
Ripeness usually needs its own source or manual labelling. For manual labelling, just sort images into class folders (a folder of `ripe/`, `unripe/`, `overripe/`) and run `split` without `--mapping`. Be consistent in your labelling rules (e.g. write down what "overripe" means for each fruit).

## 8. Installation

```bash
# Python 3.10 - 3.12 recommended
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## 9. Training Instructions

**Google Colab (recommended):** open `notebooks/training.ipynb` in Colab, switch the runtime to GPU and run the cells top to bottom.

**Local machine** (run from the project root; a GPU is strongly recommended):
```bash
python -m src.prepare_dataset check --task fruit
python -m src.train_fruit
python -m src.train_ripeness
python -m src.train_defect
```
Options: `--epochs 15` (phase 1) and `--fine-tune-epochs 10` (phase 2; `0` skips it). Pretrained ImageNet weights are downloaded on first use, so internet access is required.

What happens during training:
1. **Phase 1** - MobileNetV2 is frozen; only the new Dense/Dropout/Softmax head learns (learning rate 1e-3).
2. **Phase 2** - the top 30 layers of MobileNetV2 are unfrozen and trained gently (learning rate 1e-5).
3. `EarlyStopping` (patience 4, on validation loss) stops when validation stops improving; `ModelCheckpoint` keeps the best model.
4. Class weights compensate for unequal class sizes.
5. The best model is evaluated once on the test set.

Outputs:

| File | Purpose |
|---|---|
| `models/fruit_model.keras`, `ripeness_model.keras`, `defect_model.keras` | trained models |
| `models/<task>_classes.json` | class order used by the model (needed by the app) |
| `results/<task>_accuracy.png`, `<task>_loss.png` | training vs validation curves |
| `results/<task>_confusion_matrix.png` | test-set confusion matrix |
| `results/<task>_classification_report.csv`, `<task>_metrics.json`, `<task>_history.csv` | precision / recall / F1 and summary |

To re-evaluate a saved model: `python -m src.evaluate --task fruit`.

## 10. Running the Application
```bash
streamlit run app.py
```
Open the URL shown (usually http://localhost:8501), upload a `.jpg/.jpeg/.png/.bmp` image, and click **Run prediction**. If the models are missing, the app shows what to train instead of crashing. Copy the `models/` folder from Colab into the project first.

## 11. Evaluation Metrics (simple language)
- **Accuracy** - out of all test images, the share predicted correctly. Misleading when classes are unbalanced.
- **Precision** - when the model says "Bruised", how often is it right? (Low precision = many false alarms.)
- **Recall** - out of all truly bruised fruits, how many did the model find? (Low recall = many misses.)
- **F1-score** - one number balancing precision and recall (their harmonic mean).
- **Confusion matrix** - a table: rows = true class, columns = predicted class. The diagonal is correct predictions; off-diagonal cells show which classes get mixed up.
- **Training vs validation curves** - if training accuracy keeps rising while validation accuracy stalls or falls, the model is overfitting.
- **Confidence** - the softmax probability of the predicted class. It is not a guarantee of correctness: a model can be confidently wrong, especially on images unlike its training data.

## 12. Rule-Based Quality Assessment
`src/quality_rules.py` contains a table `QUALITY_TABLE[(defect, ripeness)] -> Good / Average / Poor`. For example healthy + ripe = Good; bruised/spotted + ripe = Average; rotten (or bruised/spotted + overripe) = Poor. Edit the table to change the rules. These rules are **not learned** and are shown separately from the deep-learning predictions in the app. A warning appears when any prediction confidence is below `LOW_CONFIDENCE_THRESHOLD` (0.60).

## 13. Expected Output
The app shows, for example (illustrative format only, not a measured result):
```text
Fruit: Apple        Confidence: xx%
Ripeness: Ripe      Confidence: xx%
Defect: Bruised     Confidence: xx%
Overall Quality (rule-based): Average
```
Actual numbers depend on your dataset and training. Record them from `results/` and report them as *experimental results*; do not quote values from anywhere else.

## 14. Project Structure

```text
fruit-quality-assessment/
├── dataset/                  your images (see section 7)
├── models/                   trained .keras models + class-name json (created by training)
├── notebooks/training.ipynb  Colab training notebook
├── src/
│   ├── config.py             paths, image size, class names
│   ├── preprocessing.py      dataset validation, loading, augmentation
│   ├── prepare_dataset.py    dataset checker and splitter
│   ├── train_common.py       shared model-building and training code
│   ├── train_fruit.py        trains Model 1
│   ├── train_ripeness.py     trains Model 2
│   ├── train_defect.py       trains Model 3
│   ├── evaluate.py           metrics, confusion matrix, curves
│   ├── prediction.py         model loading and inference
│   └── quality_rules.py      editable rule-based quality layer
├── app.py                    Streamlit web app
├── requirements.txt
├── docs/                     PBL report content, PPT outline, viva Q&A
└── results/                  graphs and metrics (created by training)
```

## 15. Limitations
- The system assesses **visible characteristics from an image only**. It cannot reliably determine internal quality, taste, sweetness, nutritional value, pesticide residue or hidden defects.
- It is **not a replacement for professional food-quality inspection**.
- Performance depends on the training data; different lighting, backgrounds, camera types and fruit varieties can reduce accuracy.
- Ripeness and defect models are trained across all fruits, so they may be less reliable for a fruit that is under-represented.
- The models always output a class, even for a non-fruit image; check the confidence and use the app only with single-fruit photos.
- "Bruised" and "spotted" can look similar, and ripeness labels are partly subjective.
- The quality rating is rule-based and only as good as its rules and the underlying predictions.
- The code was written without access to your dataset, so the training scripts have not been run on real data by the author; test them on a small dataset first.

## 16. Future Enhancements
More fruit varieties; larger, more diverse datasets; object detection for several fruits in one image; segmentation of defective regions; severity estimation; a mobile app; real-time camera detection; Grad-CAM explainability; IoT / conveyor camera integration; EfficientNetB0 comparison.

## 17. Troubleshooting
| Message | Fix |
|---|---|
| `Dataset folder not found` / `Missing folder` | Follow the layout in section 7 exactly. |
| `missing class folders` | Create every class folder in `train`, `validation` and `test`. |
| `Trained model files not found` | Run the three training commands, then restart the app. |
| `ModuleNotFoundError: src` | Run commands from the project root, using `python -m src.<name>`. |
| Training very slow | Use Colab with a GPU runtime. |
