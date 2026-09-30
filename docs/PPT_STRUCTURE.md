# PPT Presentation Structure (about 15 slides, 10-12 minutes)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Project title, your name, roll number, guide, department, college, date |
| 2 | Problem statement | Manual fruit inspection is slow and subjective; need a fast, consistent visual assessment |
| 3 | Objectives | Fruit type, ripeness, defect, overall quality, Streamlit demo |
| 4 | Existing vs proposed system | Two-column comparison (manual / classical ML vs transfer-learning pipeline) |
| 5 | System architecture | Block diagram: image -> preprocessing -> 3 models -> rule layer -> dashboard |
| 6 | Dataset | Sources, classes, counts per split **[your actual numbers]**, folder layout, how leakage was prevented |
| 7 | Preprocessing and augmentation | Resize, scaling, flip, rotation, zoom, brightness/contrast; one before/after image |
| 8 | Model architecture | MobileNetV2 -> GAP -> Dense -> Dropout -> Softmax; why transfer learning; why 3 models |
| 9 | Training strategy | Two-phase training, Adam, cross-entropy, early stopping, checkpoint, class weights |
| 10 | Evaluation metrics | Accuracy, precision, recall, F1, confusion matrix in one line each |
| 11 | Results | Metrics table + accuracy/loss graphs + confusion matrix **[your actual results only]** |
| 12 | Rule-based quality layer | Rule table; stress that it is NOT a deep-learning prediction |
| 13 | Live demo / screenshots | Upload -> predictions with confidence -> quality rating; show one failure case too |
| 14 | Limitations and future scope | Visible features only; dataset dependence; Grad-CAM, detection, segmentation, mobile app |
| 15 | Conclusion and Q&A | Two or three takeaways; thank you |

**Tips:** keep one idea per slide; use screenshots of your real output; prepare one wrong prediction and be ready to explain why; keep a backup of the demo images in case the projector or internet fails.
