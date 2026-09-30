# PBL Report Content
**Title:** Deep Learning-Based Fruit Quality, Ripeness and Defect Assessment

> How to use: copy each section into your report and edit it in your own words. Sections marked **[FILL]** need your real data. Do not copy any numbers from elsewhere; only report values from your own training run (`results/`). Add and cite only papers you have actually read.

---

## 1. Abstract
Assessing fruit quality by eye is slow and subjective. This project presents a deep-learning-based system that analyses a single fruit image and predicts the fruit type, ripeness level and visible defect, together with a confidence score for each prediction. Three separate convolutional neural network classifiers were built using transfer learning with MobileNetV2 pretrained on ImageNet. The models were trained on a labelled image dataset **[FILL: dataset source and size]** with data augmentation and evaluated using accuracy, precision, recall, F1-score and confusion matrices. A transparent rule-based layer then combines the ripeness and defect predictions into an overall quality rating (Good / Average / Poor). The system is deployed as a Streamlit web application. Results: **[FILL: one sentence with your actual test accuracies]**. The system assesses only visible characteristics and does not replace professional inspection.

**Keywords:** deep learning, transfer learning, MobileNetV2, fruit quality, ripeness classification, defect detection, Streamlit.

## 2. Introduction
Fruits are graded for freshness, ripeness and defects before sale. This is usually done by trained people looking at each fruit, which can be tiring and inconsistent. Computer vision and deep learning can automate part of this by learning visual patterns from labelled images. Convolutional neural networks (CNNs) are especially good at this because they automatically learn features such as colour, texture and shape. Training a CNN from scratch needs large data and computing power, so this project uses transfer learning: a network pretrained on a large general image dataset is reused and adapted to fruit images.

## 3. Problem Statement
To design and implement a deep-learning-based system that, given an image of a fruit, predicts (a) the fruit type, (b) its ripeness level, (c) its visible defect, and (d) an overall quality rating, in a form that is easy to demonstrate through a web interface.

## 4. Objectives
1. Build a fruit-type classifier (Apple, Banana, Orange, Mango).
2. Build a ripeness classifier (Unripe, Ripe, Overripe).
3. Build a defect classifier (Healthy, Bruised, Spotted, Rotten).
4. Apply transfer learning and data augmentation to obtain good results with limited data and compute.
5. Evaluate the models with standard metrics on unseen test images.
6. Provide a rule-based overall quality assessment, clearly separated from the learned predictions.
7. Develop a simple Streamlit application for demonstration.

## 5. Literature Survey
> Write this section from papers you have actually read. The points below give the general background; replace or extend them with your own reviewed papers and cite them. Check every reference detail before submission.

| Topic | Summary (general background) | Reference from list below |
|---|---|---|
| Computer vision for fruit/vegetable quality | Reviews describe how image-based methods have been used to grade fruits by colour, size, shape and defects, and note challenges such as lighting variation and natural variability. | [1] |
| Classical machine learning | Earlier systems extracted hand-designed features (colour histograms, texture, shape) and used classifiers such as SVM or k-NN. These need careful feature design and generalise less well. | [1] |
| CNNs for fruit recognition | Deep CNNs learn features directly from pixels; a public fruit-image dataset (Fruits-360) has been used in CNN-based fruit recognition work. | [2] |
| Transfer learning | Reusing features learned on a large source dataset helps when the target dataset is small. | [3], [4] |
| Efficient architectures | MobileNet-family and EfficientNet models were designed to give good accuracy with small size and low computation, which suits Colab and deployment on modest hardware. | [5], [6] |
| Augmentation and regularisation | Augmentation and dropout are common ways to reduce overfitting. | [7], [8] |
| Explainability | Grad-CAM highlights the image regions that influence a CNN's decision (planned as future work). | [9] |

**Research gap addressed here:** many demonstrations predict only one attribute. This project combines fruit type, ripeness and defect predictions with a transparent rule-based quality layer in a simple web app.

## 6. Existing System
- **Manual inspection:** people judge colour, firmness and blemishes visually. It is subjective, slow and varies from person to person.
- **Classical image processing / ML:** hand-crafted features with SVM/k-NN or thresholding; sensitive to lighting and background and hard to extend to new fruits.
- **Single-task deep-learning demos:** usually predict only fresh vs rotten or only the fruit name.

## 7. Proposed System
An end-to-end pipeline: the user uploads an image; OpenCV decodes and resizes it; three MobileNetV2-based classifiers predict fruit type, ripeness and defect with confidence; a rule table converts ripeness and defect into an overall quality rating; the Streamlit app displays everything. Advantages: transfer learning keeps training feasible on free Colab; separate models are easy to evaluate and improve; rules are transparent and editable.

## 8. Methodology
1. **Data collection and labelling** into `dataset/<task>/<split>/<class>/`.
2. **Splitting** into train/validation/test before augmentation; duplicate check to avoid leakage.
3. **Preprocessing:** resize to 224x224; scale pixels to [-1, 1].
4. **Augmentation (training only):** horizontal flip, rotation, zoom, small brightness and contrast changes.
5. **Model building:** MobileNetV2 (ImageNet weights) + Global Average Pooling + Dense(128) + Dropout(0.3) + Softmax.
6. **Training:** Phase 1 trains the new head with the base frozen; Phase 2 fine-tunes the top 30 base layers at a low learning rate. Loss: sparse categorical cross-entropy; optimiser: Adam; metric: accuracy; early stopping and checkpointing on validation loss; class weights for imbalance.
7. **Evaluation** on the test set.
8. **Rule-based quality assessment.**
9. **Deployment** with Streamlit.

## 8.1 Preprocessing steps and why
| Step | Reason |
|---|---|
| Resize to 224x224 | MobileNetV2 expects a fixed input size. |
| Pixel scaling to [-1, 1] | Matches the range the pretrained network was trained with; helps stable training. |
| Horizontal flip | A fruit looks the same mirrored; gives more variety. |
| Rotation | Fruits can be photographed at any angle. |
| Zoom | Distance from the camera varies. |
| Brightness/contrast | Lighting varies. Kept mild, and hue is not changed, because colour is the key cue for ripeness. |

## 9. System Architecture
(Insert the diagram from README section 6, redrawn as a block diagram.)

```text
Image -> OpenCV preprocessing -> [Fruit CNN | Ripeness CNN | Defect CNN]
      -> Rule-based quality layer -> Streamlit dashboard
```

## 10. Dataset Description
**[FILL]** Source(s) of images (public dataset names with links you verified, and/or self-collected), licence, number of images per class and per split (a table such as below), image conditions (background, lighting, camera), labelling method and who labelled, and how duplicates were removed.

| Task | Class | Train | Validation | Test |
|---|---|---|---|---|
| Fruit | apple / banana / mango / orange | [FILL] | [FILL] | [FILL] |
| Ripeness | unripe / ripe / overripe | [FILL] | [FILL] | [FILL] |
| Defect | healthy / bruised / spotted / rotten | [FILL] | [FILL] | [FILL] |

## 11. Model Architecture
| Layer | Output | Purpose |
|---|---|---|
| Input | 224x224x3 | RGB image |
| Rescaling | 224x224x3 | pixels to [-1, 1] |
| MobileNetV2 (pretrained, top removed) | 7x7x1280 | feature extraction |
| GlobalAveragePooling2D | 1280 | one value per feature map; fewer parameters than Flatten |
| Dense(128, ReLU) | 128 | learns task-specific combinations |
| Dropout(0.3) | 128 | randomly switches off units to reduce overfitting |
| Dense(N, Softmax) | N | class probabilities (N = 4, 3, 4) |

**Why MobileNetV2:** small, fast, uses depthwise separable convolutions, works well with transfer learning on modest hardware.

## 12. Algorithm
```text
1. Load images from dataset/<task>/{train,validation,test}
2. Validate structure; check duplicates across splits
3. For each task in {fruit, ripeness, defect}:
   a. Build MobileNetV2 + classification head; freeze base
   b. Train head with augmentation, class weights, early stopping, checkpoint
   c. Unfreeze top layers; fine-tune with low learning rate
   d. Keep the model with the lowest validation loss
   e. Evaluate on the test set; save graphs and metrics
4. For a new image:
   a. Decode, convert to RGB, resize to 224x224
   b. Run the three models; take the class with highest probability and its probability as confidence
   c. Look up (defect, ripeness) in the rule table -> Good / Average / Poor
   d. Display results
```

## 13. Implementation
Language: Python. Libraries: TensorFlow/Keras (models and training), OpenCV (image decoding/resizing), NumPy, Pandas, Matplotlib (graphs), Scikit-learn (metrics, class weights), Streamlit (web app). Training on Google Colab GPU. Describe the module roles from README section 14. Include screenshots **[FILL]** of the folder structure, training output, and the app.

## 14. Results
> Only actual results from your own runs go here.

**14.1 Experimental setup [FILL]:** hardware, epochs run, batch size 32, image size 224, learning rates 1e-3 / 1e-5, random seed 42.

**14.2 Test-set metrics [FILL from `results/<task>_metrics.json`]**
| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Fruit | [FILL] | [FILL] | [FILL] | [FILL] |
| Ripeness | [FILL] | [FILL] | [FILL] | [FILL] |
| Defect | [FILL] | [FILL] | [FILL] | [FILL] |

**14.3 Figures [FILL]:** insert `results/<task>_accuracy.png`, `<task>_loss.png`, `<task>_confusion_matrix.png`.

**14.4 Discussion [FILL]:** Which classes are confused? Any sign of overfitting? What would you improve? Also mention the *expected* behaviour separately (e.g. "we expect bruised vs spotted to be confused because they look similar") and label it as expectation, not result.

## 15. Advantages
Fast and consistent; no manual feature design; transfer learning needs less data and compute; modular (each model can be improved separately); transparent rule layer; simple web interface; low-cost (needs only a camera).

## 16. Limitations
Only visible characteristics are assessed (not internal quality, taste, nutrition, pesticide residue or hidden defects); depends on dataset quality and variety; sensitive to lighting/background changes; single fruit per image; four fruits only; subjective ripeness labels; always predicts one of the known classes even for unknown objects; rule-based quality is simplistic; not a substitute for professional food-quality inspection.

## 17. Applications
Educational demonstration; supermarket or home quick check; sorting support in small packing units; input stage for larger quality-control systems; base for mobile apps that help consumers.

## 18. Future Scope
More fruit varieties; larger and more diverse datasets; object detection for multiple fruits; segmentation of defective regions and severity estimation; mobile app; real-time camera detection; Grad-CAM explainability; IoT/conveyor-belt camera integration; comparing EfficientNetB0 with MobileNetV2.

## 19. Conclusion
The project implements a practical pipeline for image-based assessment of fruit type, ripeness and visible defects using transfer learning, and presents the results in an easy-to-use web application with a clearly separated rule-based quality rating. **[FILL: one or two sentences summarising your actual results.]** The system is limited to visible characteristics and is intended for learning and demonstration, not as a replacement for professional inspection.

## 20. References
> Verify each entry (title, venue, year, pages/DOI) from the original source before submitting, and remove any you have not read.

1. Bhargava, A., & Bansal, A. "Fruits and vegetables quality evaluation using computer vision: A review." *Journal of King Saud University - Computer and Information Sciences.*
2. Mureşan, H., & Oltean, M. "Fruit recognition from images using deep learning." *Acta Universitatis Sapientiae, Informatica*, 2018.
3. Pan, S. J., & Yang, Q. "A Survey on Transfer Learning." *IEEE Transactions on Knowledge and Data Engineering*, 2010.
4. Deng, J., et al. "ImageNet: A large-scale hierarchical image database." *CVPR*, 2009.
5. Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*, 2018.
6. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, 2019.
7. Shorten, C., & Khoshgoftaar, T. M. "A survey on image data augmentation for deep learning." *Journal of Big Data*, 2019.
8. Srivastava, N., et al. "Dropout: A simple way to prevent neural networks from overfitting." *Journal of Machine Learning Research*, 2014.
9. Selvaraju, R. R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV*, 2017.
10. Kingma, D. P., & Ba, J. "Adam: A method for stochastic optimization." *ICLR*, 2015.
11. Abadi, M., et al. "TensorFlow: A system for large-scale machine learning." *OSDI*, 2016.
