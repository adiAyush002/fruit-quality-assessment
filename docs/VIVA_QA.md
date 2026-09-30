# Viva Questions and Answers

**1. What is your project about?**
It analyses a fruit image with deep learning and predicts the fruit type, ripeness and visible defect with confidence scores, then gives a rule-based overall quality rating.

**2. Why deep learning instead of traditional image processing?**
Traditional methods need hand-designed features (colour thresholds, texture) that break when lighting or fruit variety changes. CNNs learn features automatically from examples.

**3. What is a CNN?**
A neural network that uses convolution filters to detect local patterns (edges, textures) and combines them layer by layer into higher-level features, followed by a classifier.

**4. What is transfer learning and why did you use it?**
Reusing a network pretrained on a large dataset (ImageNet) for a new task. It needs less data, trains faster and usually generalises better than training from scratch, which is important on free Colab.

**5. Why MobileNetV2?**
It is small and fast (depthwise separable convolutions) yet accurate, and works well with transfer learning on limited hardware. EfficientNetB0 is a possible alternative.

**6. What is Global Average Pooling?**
It averages each feature map into one number, turning 7x7x1280 into 1280 values. It has no parameters, reduces overfitting and is simpler than Flatten.

**7. What is Dropout?**
During training, a fraction of neurons (30% here) is randomly switched off so the network cannot rely on specific neurons; this reduces overfitting. It is inactive at prediction time.

**8. Why softmax and cross-entropy?**
Softmax turns outputs into probabilities that sum to 1 for single-label multi-class classification. Cross-entropy penalises low probability on the correct class. We use the sparse version because labels are integers.

**9. Why three separate models?**
Each question uses different visual cues, can use different datasets, and is easier to train, evaluate and debug separately. Drawback: three times the storage and inference.

**10. Why not one multi-output model?**
It is possible, but it needs images labelled for all three attributes at once, which is hard to obtain from public datasets. Separate models keep the dataset requirements simple.

**11. What is data augmentation and why do you use it?**
Creating modified copies of training images (flip, rotation, zoom, brightness/contrast) on the fly to increase variety and reduce overfitting. It is applied only to training data.

**12. Why did you not change hue or saturation?**
Colour is the main cue for ripeness and defects; altering hue could turn a ripe fruit into an "unripe-looking" one and confuse the labels.

**13. What is data leakage and how did you prevent it?**
When the same or nearly the same images appear in training and test sets, making results look better than reality. We split before augmentation, augment only training data, remove exact duplicates and check duplicates across splits with a script. Near-duplicates need manual care.

**14. Why do you need training, validation and test sets?**
Training data teaches the model; validation data is used during training for early stopping and checkpointing; the test set is used once at the end for an unbiased estimate.

**15. What are overfitting and underfitting?**
Overfitting: high training accuracy but poor validation/test accuracy (memorised data). Underfitting: poor on both. Remedies: augmentation, dropout, early stopping, more data.

**16. Explain the two-phase training.**
Phase 1 trains only the new head with the pretrained base frozen. Phase 2 unfreezes the top 30 layers of the base and trains with a very small learning rate (1e-5) to adapt features to fruit images without destroying pretrained knowledge.

**17. What is early stopping and checkpointing?**
Early stopping ends training when validation loss stops improving for 4 epochs and restores the best weights. Checkpointing saves the best model to disk.

**18. Why use class weights?**
If some classes have fewer images, the model may ignore them. Class weights make errors on rare classes count more.

**19. Explain precision, recall and F1.**
Precision: of the images predicted as class X, how many really are X. Recall: of the images that truly are X, how many were found. F1: harmonic mean of both. Accuracy alone can hide poor performance on small classes.

**20. What does the confusion matrix show?**
Rows are true classes, columns are predicted classes; the diagonal shows correct predictions and off-diagonal cells show which classes are confused.

**21. What does "confidence" mean? Is 95% confidence always correct?**
It is the softmax probability of the predicted class. No: models can be confidently wrong, especially for unfamiliar images (e.g. a non-fruit photo).

**22. Is the quality score a deep-learning output?**
No. It is a rule table in `quality_rules.py` combining the ripeness and defect predictions. It is shown separately and can be edited without retraining.

**23. What accuracy did you get?**
State only your real test results from `results/`, and say which dataset they refer to. Do not quote numbers you did not measure. Add that results may not transfer to different cameras or lighting.

**24. What are the limitations?**
Only visible features; cannot judge internal quality, taste, nutrition, pesticide residue or hidden defects; depends on data; four fruits; one fruit per image; not a replacement for professional inspection.

**25. What happens if a non-fruit image is uploaded?**
The models still output one of their known classes. The app shows confidences and a low-confidence warning below 60%, but it cannot reliably detect out-of-scope images; adding an "unknown/not fruit" class or out-of-distribution detection is future work.

**26. Why is the image resized to 224x224?**
MobileNetV2's pretrained weights expect that size, and a fixed size lets images be batched.

**27. Why scale pixels to [-1, 1]?**
It matches the input range MobileNetV2 was trained with; the scaling is built into the model, so the app just passes raw 0-255 pixels.

**28. How would you improve the system?**
More and more varied data, more fruits, Grad-CAM to see what the model looks at, segmentation of defects, severity estimation, object detection for multiple fruits, mobile or real-time camera deployment.

**29. What is Grad-CAM?**
A method that produces a heat map showing which image regions most influenced a CNN's prediction, useful for checking that the model looks at the fruit and not the background.

**30. How do you run your project?**
Place the dataset in `dataset/`, train with `python -m src.train_fruit`, `train_ripeness`, `train_defect` (or the Colab notebook), then run `streamlit run app.py`.
