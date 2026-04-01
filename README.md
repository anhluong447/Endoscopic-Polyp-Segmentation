# PolySeg: ResUNet for Endoscopic Polyp Segmentation

> **Note:** This project was developed, trained, and executed entirely within the **Kaggle** notebook environment. So if you wanna test it out, I'd recommend you go directly to my Kaggle code, and run it by all mean (Oh, don't forget to setup your own Wandb key). You can find it on: https://www.kaggle.com/code/sarahhimeko/polyseg-resunet-for-endoscopic-polyp-segmentation.

## 📌 Overview
<p align="center">
  <img width="1560" height="676" alt="Polyps" src="https://github.com/user-attachments/assets/b056a04c-58ad-411c-915b-5c13beda6129" />
  <br>
  <i>Fig 1. PolySeg Pipeline.</i>
</p>

PolySeg is a deep learning pipeline designed for the automated segmentation of polyps in endoscopic imagery. The system utilizes a custom-built Residual UNet (Res-UNet) architecture implemented in PyTorch. A major focus of this project is mitigating severe medical data class imbalance through a carefully weighted custom loss function.

## ⚙️ Core Architecture & Technologies
* **Framework:** PyTorch (`torch`, `torch.nn`).
* **Model Architecture:** A custom `UNet` integrating `encoder_block`, `res_encoder_block`, `decoder_block`, and `res_decoder_block` modules, bridged by a central `ResidualBlock` bottleneck.
* **Custom Loss Function:** Implements `CEDiceLoss`, a hybrid function combining Cross-Entropy and Dice Loss. It explicitly penalizes class imbalance using a rigid weight distribution of `[0.4, 0.55, 0.05]`.
* **Data Augmentation:** Leverages the `albumentations` library for dynamic pipeline transformations, including `HorizontalFlip`, `VerticalFlip`, `RandomGamma`, and `RGBShift`.
* **Experiment Tracking:** Integrated directly with **Weights & Biases (WandB)** to monitor real-time training and validation losses.

## 🚀 Pipeline Workflow
1. **Data Loading & Preprocessing:** Reads images and masks from the `bkai-igh-neopolyp` Kaggle dataset. Normalizes inputs and thresholds masks into 3 distinct classes.
2. **Training:** The model is trained over 30 epochs using the Adam optimizer (`lr=1e-04`) and a `StepLR` scheduler (`gamma=0.6`, `step_size=4`).
3. **Inference & Visualization:** Generates and saves predicted masks using `torch.argmax` and `F.one_hot` encoding.
4. **Validation Export:** Automatically processes the predicted masks into Run-Length Encoding (RLE) strings, outputting an `output.csv` file formatted for competition submission.

## 🏆 Performance & Leaderboard Results

Since this pipeline was developed for a competitive Kaggle environment, the model's true generalization performance was evaluated against an unseen, unannotated test set by the Kaggle scoring engine.

* **Evaluation Metric:** Mean Dice Coefficient / Intersection over Union (IoU)
* **Leaderboard Score:** **0.84261**
* **Ranking/Standing:** **Rank 29**

While testing locally, the custom `CEDiceLoss` proved highly effective at forcing the Res-UNet to recognize minority polyp pixels, demonstrating stable convergence without severe overfitting thanks to aggressive `albumentations` policies.

## 📊 Visualization Results
<p align="center">
  <img width="891" height="990" alt="__results___63_0" src="https://github.com/user-attachments/assets/e22972c6-71ed-4107-a80d-c0b514a39450" />
  <br>
  <i>Fig 2. Model Inference: Original Endoscopic Image vs. Predicted Segmentation Mask.</i>
</p>

Visualizing the model's predictive capabilities on unseen test data, demonstrating the effectiveness of the custom Res-UNet and CEDiceLoss in isolating polyp regions.

## 📂 Kaggle Directory Structure
The data loaders are explicitly configured for the Kaggle environment paths:
* **Training Data:** `/kaggle/input/bkai-igh-neopolyp/train/train/`
* **Training Ground Truth:** `/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt/`
* **Testing Data:** `/kaggle/input/bkai-igh-neopolyp/test/test/`
* **Outputs:** Predicted masks are saved to `/kaggle/working/predicted_masks/` and the final submission to `/kaggle/working/output.csv`.
