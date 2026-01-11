# Evidence YOLO Model Based on VFM Distillation and Distribution Regression Modeling

## Project Introduction

This project implements an Evidence YOLO model combining Visual Feature Matching (VFM) distillation and Distribution Regression Modeling based on the YOLO framework. By introducing evidence theory and knowledge distillation techniques, this model improves the accuracy and uncertainty estimation capability of object detection.

## Core Technologies

### 1. Evidence YOLO Model

The Evidence YOLO model is an object detection model based on evidence theory, which improves the reliability of detection results by modeling uncertainty. It extends YOLOv8 with the following main improvements:

- **Evidence Regression**: Uses Normal Inverse Gamma (NIG) distribution to model the uncertainty of bounding boxes
- **Evidential Deep Regression Loss (EDRLoss)**: Combines negative log-likelihood and regularization terms to optimize evidence parameters
- **Uncertainty Awareness**: Can output confidence and uncertainty information of detection results

### 2. VFM Distillation (Visual Feature Matching Distillation)

VFM distillation is a knowledge distillation technique that improves the performance of student models by matching the visual features of teacher and student models. This project uses **DINOv3** as the Vision Foundation Model (VFM) as the teacher model.

#### DINOv3 Teacher Model

DINOv3 is the latest generation of self-supervised visual Transformer model developed by Facebook Research, with strong feature representation capabilities. This project uses the ViT-B/16-LVD-1689M variant of DINOv3 as the teacher model, which has the following characteristics:

- **Large-scale Pre-training**: Pre-trained on 16.89M labeled LVD (Long Video Dataset)
- **Efficient Architecture**: Based on ViT-B/16 backbone network, balancing performance and computational efficiency
- **Powerful Feature Representation**: Can extract semantically rich, hierarchical visual features
- **Self-supervised Learning**: Can learn high-quality visual representations without manual annotations

The DINOv3 teacher model implementation is located in the `ultralytics/nn/teacher_models.py` file, which loads pre-trained weights via `torch.hub` and uses its output feature maps for distillation training.

#### VFM Distillation Features

- **Cosine Similarity Loss**: Calculates distillation loss based on cosine similarity of feature maps
- **Feature Alignment**: Automatically aligns feature maps of different sizes
- **Anti-NaN Protection**: Prevents numerical instability by adding a small value
- **Adaptive Scaling**: Dynamically adjusts distillation loss weight based on IoU

### 3. Distribution Regression Modeling (EDR Modeling)

Distribution Regression Modeling, namely Evidential Deep Regression (EDR) Modeling, improves detection accuracy and estimates uncertainty by learning target distributions. It mainly includes:

- **Evidential Deep Regression Loss (EDRLoss)**: Uses Normal Inverse Gamma (NIG) distribution to model bounding box uncertainty
- **Multi-scale Feature Fusion**: Combines feature maps of different scales to improve detection performance

## Code Structure

```
distillation/
├── ultralytics/          # Core YOLO framework code
│   ├── models/           # Model definitions
│   ├── utils/            # Utility functions
│   │   ├── loss.py       # Loss function definitions (including EDRLoss and DistillationLoss)
│   ├── nn/               # Neural network modules
│   │   ├── modules/      # Network layer modules
│   │   │   ├── head.py   # Detection head definition
├── datasets/             # Dataset directory
│   ├── VOC/              # VOC dataset
│   ├── Apple/            # Apple dataset
├── train.py              # Training script
├── val.py                # Validation script
```

## Key Components Detailed

### Evidential Deep Regression Loss (EDRLoss)

EDRLoss is an evidential theory-based regression loss used to model the uncertainty of bounding boxes. It represents the probability distribution of bounding boxes through the Normal Inverse Gamma (NIG) distribution and optimizes the following parameters:

- `gamma`: Mean parameter
- `v`: Precision parameter
- `alpha`: Shape parameter
- `beta`: Scale parameter

#### EDRLoss Formula

EDRLoss consists of three parts:

1. **Negative Log-Likelihood (NLL)**: Measures the matching degree between predicted distribution and true values
   
   ```math
   NLL = 0.5 \log(\pi / v) - \alpha \log(2\beta(1 + v)) + (\alpha + 0.5) \log(v(y - \gamma)^2 + 2\beta(1 + v)) + \Gamma(\alpha) - \Gamma(\alpha + 0.5)
   ```

2. **Regularization Term**: Prevents overfitting
   
   ```math
   RegLoss = \lambda |y - \gamma| (2v + \alpha)
   ```

3. **Uncertainty Constraint**: Encourages the model to output reasonable uncertainty estimates
   
   ```math
   UCLoss = (y - \gamma)^2 \frac{v(\alpha - 1)}{\beta(1 + v)}
   ```

4. **Total Loss**:
   
   ```math
   EDRLoss = NLL + RegLoss + UCLoss
   ```

Where, $y$ is the true value, $\gamma, v, \alpha, \beta$ are the NIG distribution parameters predicted by the model, $\lambda$ is the regularization coefficient, and $\Gamma(\cdot)$ is the gamma function.

### Distillation Loss

DistillationLoss implements feature distillation based on cosine similarity, mainly including the following steps:
1. Align the feature map sizes of teacher and student models
2. Flatten the feature maps and perform L2 normalization
3. Calculate cosine similarity
4. Calculate distillation loss (1 - cosine similarity)

### Detection Loss (v8DetectionLoss)

v8DetectionLoss integrates multiple loss functions, including:
- Bounding box loss (IoU loss)
- Classification loss (BCE loss)
- Distribution focal loss (DFLoss)
- Evidential deep regression loss (EDRLoss)
- Distillation loss (DistillationLoss)

## Quick Start

### Environment Configuration

This project is based on Anaconda environment and uses PyTorch framework. Please ensure the following dependencies are installed:

- Python 3.10+
- PyTorch 2.4+
- Ultralytics YOLO
- NumPy 1.24.0-1.26.4
- Matplotlib

### Train the Model

Use the following command to train the Evidence YOLO model:

```python
python train.py
```

Training configuration can be modified in the `train.py` file, the main parameters include:

- `epochs`: Number of training epochs
- `img_size`: Input image size
- `pretrained`: Whether to use pre-trained weights
- `project_name`: Project name
- `device`: Training device
- `distillation`: Whether to enable knowledge distillation

### Validate the Model

Use the following command to validate the model performance:

```python
python val.py
```

## Configuration Instructions

### Distillation Configuration

During training, knowledge distillation can be enabled by setting `distillation=True`. Distillation-related hyperparameters can be modified in `ultralytics/cfg/default.yaml`:

- `dis`: Distillation loss weight
- `distillation`: Whether to enable distillation

### Evidence Regression Configuration

Evidence regression-related configurations can be modified in the model definition, the main parameters include:

- `evidence`: Whether to enable evidence regression
- `reg_max`: Maximum range of distribution regression
- `edr`: Evidence regression loss weight

## Dataset Support

This project supports multiple datasets, including:

- VOC dataset
- Apple dataset
- COCO dataset (need to be prepared by yourself)

Dataset configuration files are located in the `ultralytics/cfg/datasets/` directory.

## Model Performance

The Evidence YOLO model based on VFM distillation and distribution regression modeling has shown excellent performance on multiple datasets:

- Improved detection accuracy (mAP)
- Enhanced uncertainty estimation capability
- Reduced false detection rate
- Improved small target detection performance

## Application Scenarios

This model is suitable for the following scenarios:

- **High-precision object detection**: Application scenarios requiring high accuracy
- **Uncertainty awareness**: Applications that need to know the reliability of detection results
- **Resource-constrained devices**: Reduce model size while maintaining performance through distillation technology
- **Few-shot learning**: Learn from a small amount of labeled data using distillation technology

## References

1. YOLOv8: https://github.com/ultralytics/ultralytics
2. RT-DETRv4: https://github.com/lyuwenyu/RT-DETRv4
3. Evidence-Based Deep Learning for Computer Vision: https://arxiv.org/abs/1806.01768
4. Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
5. DINOv3: https://github.com/facebookresearch/dinov3

## License

This project is based on the AGPL-3.0 license, please refer to the LICENSE file for details.

## Contact Information

If you have any questions or suggestions, please contact us through the following ways:

- Project address: https://github.com/yourusername/evidence-yolo
- Email: your.email@example.com
