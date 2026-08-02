# Cats vs Dogs Image Classifier
A Convolutional Neural Network that classifies photos as either a cat or a dog — built as a final capstone project for an ML internship, with a live Flask demo deployed on Vercel for real-time predictions on new images.

## Objective
Build a CNN that classifies real-world cat and dog photos accurately enough to generalize to genuinely new images, not just the ones it was trained on — including diagnosing and fixing overfitting once it appeared, and deploying the final model as a usable public demo.

## Dataset
The **Kaggle Cats vs Dogs** dataset — 24,991 labeled images across two classes (Cat, Dog), after removing a small number of corrupted/unreadable files during preprocessing.
- Split 80/20 into 19,993 training and 4,998 validation images
- Resized to 128×128 pixels
- Pixel values rescaled from `[0, 255]` to `[0, 1]`

## Model Architecture
A CNN with three convolutional blocks, built in TensorFlow/Keras:
```
Input (128, 128, 3)
→ Conv2D(32) → MaxPooling2D
→ Conv2D(64) → MaxPooling2D
→ Conv2D(128) → MaxPooling2D
→ Flatten
→ Dense(128, relu) → Dropout(0.5)
→ Dense(1, sigmoid)
```
Compiled with binary cross-entropy loss and the Adam optimizer.

## The Overfitting Fix
A baseline model (10 fixed epochs, no regularization) reached 82.3% validation accuracy — but training accuracy climbed to 98.7% while validation loss more than doubled, a clear overfitting signature. Three fixes were applied and the model retrained:
- **Data augmentation** (random flip, rotation, zoom) as a model layer, active only during training
- **Dropout(0.5)** before the output layer
- **EarlyStopping** (monitoring validation loss, patience 3, restoring best weights) instead of a fixed epoch count

| Metric | Baseline | Regularized (final) |
|---|---|---|
| Training stopped at | Epoch 10 (fixed) | Epoch 13 (early stopping) |
| Training accuracy | 98.7% | ~85.7% |
| Validation accuracy | 82.3% | **86.7%** |
| Validation loss (best) | 0.91 (rising) | **0.306 (stable)** |
| Train/val gap | ~16 points, widening | ~1 point, stable |

Regularization improved accuracy *and* generalization at the same time — the retrained model is both more accurate and more trustworthy than the original.

## Real-World Validation
Beyond the held-out validation set, the final model was tested on two freshly downloaded photos never seen during training:
- A cat photo → correctly predicted **Cat**, 80.7% confidence
- A dog photo → correctly predicted **Dog**, 99.2% confidence

Notably, the *overfit* baseline model had been 99.999% confident on the same cat photo — a sign of memorization, not genuine certainty. The regularized model's more modest confidence score is a more honest reflection of what the model actually knows.

## Live Demo
The trained model is converted to TensorFlow Lite and deployed as a Flask app on Vercel — upload any photo and get a real-time Cat/Dog prediction with a confidence score.

```bash
pip install -r requirements.txt
python app.py
```

The app is split into two files: `model.py` loads the `.tflite` model and handles inference, and `app.py` defines the Flask routes and serves the frontend.

## Future Improvements
- Use a truly held-out third test set, separate from the validation set used for early stopping
- Apply transfer learning with a pretrained backbone (e.g., MobileNetV2) for likely higher accuracy with less training time
- Add a confusion matrix and per-class precision/recall to see whether errors skew toward one class

## Repository Structure
```
├── cats_dogs_model.tflite                                 # Saved trained model, converted to TFLite
├── model.py                                               # Loads the TFLite model and runs predictions
├── app.py                                                 # Flask app: routes and frontend
├── templates/
│   └── index.html                                          # Upload UI
├── vercel.json                                            # Vercel function configuration
├── requirements.txt
└── README.md
```

## Tech Stack
`tensorflow` / `keras` · `flask` · `ai-edge-litert` · `pillow` · `numpy` · `matplotlib` · `vercel`

## Author
**Ahmed Tareq** — ML Internship Final Capstone Project
