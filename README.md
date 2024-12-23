# Multiclass Object Classification in Autonomous Driving

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team Members](#team-members)
3. [Project Description](#project-description)
4. [Dataset Links](#dataset-links)
5. [Technical Details](#technical-details)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Results & Performance](#results--performance)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License & Contact](#license--contact)

## Project Overview

### Quick Start
```bash
git clone https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git
cd Multiclass-Object-Classification-in-Autonomous-Driving
pip install -r requirements.txt
```

### Project Goals
- Develop a robust object classification system for autonomous driving
- Implement and compare 8 different classification models
- Create a deployable solution with real-time inference capabilities
- Provide comprehensive model comparison and analysis

### Classification Categories
1. **Multi-class Classification** (DenseNet-121 Specialized)
   - Human (Pedestrian, Cyclist)
   - Vehicle (Car, Truck, Van, Tram)

## Team Members

### Project Team
1. Bhushan Asati
   - Role: Data Scientist
   - Models: DenseNet121, MobileNetV2 
   - Contributions: Data preprocessing, Feature Engineering, Model training, Model Optimiztion, Deployment

2. Rujuta Dabke
   - Role: Data Scientist
   - Models: EfficientNet, Faster R-CNN
   - Contributions: Feature engineering, Visualization

3. Suyash Madhavi
   - Role: Data Scientist
   - Models: Inception-v3, ResNet50
   - Contributions: Model optimization, Performance analysis

4. Anirudh Sharma
   - Role: Data Scientist
   - Models: XGBoost, Random Forest
   - Contributions: Traditional ML implementation, Evaluation metrics

## Project Description

### Project Scope
- Implementation of 8 different classification models
- Real-time inference capabilities
- Web interface for demonstration
- Comprehensive performance analysis
- Model comparison framework

### Expected Outputs
1. **Technical Deliverables**
   - Trained model weights
   - API for inference
   - Web interface
   - Performance metrics

2. **Documentation**
   - Technical documentation
   - API documentation
   - User guides
   - Performance reports

### Innovation Points
- Multi-model comparison framework
- Specialized DenseNet-121 implementation
- Hybrid classification approach
- Comprehensive evaluation metrics

## Dataset Links

- Kaggle dataset: https://www.cvlibs.net/datasets/kitti/

- KITTI dataset: https://www.kaggle.com/datasets/garymk/kitti-3d-object-detection-dataset

## Technical Details

### Project Structure
```
Multiclass-Object-Classification-KITTI/
├── data/
│   ├── raw/               
│   ├── processed/        
│   └── samples/           
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_model_evaluation.ipynb
├── models/
│   ├── resnet50_model.h5
│   ├── efficientnetb0_model.h5
│   └── ...
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── streamlit_app/
│   └── app.py
└── docker/
    └── Dockerfile
```

### Models Implemented

1. **ResNet50**
   - Architecture: Deep residual network
   - Features: Skip connections, Batch normalization
   - Performance: 98% accuracy

2. **EfficientNetB0**
   - Architecture: Compound scaling
   - Features: Balanced scaling, Optimized architecture
   - Performance: 16% accuracy

3. **MobileNetV2**
   - Architecture: Lightweight CNN
   - Features: Inverted residuals, Linear bottlenecks
   - Performance: 99% accuracy

4. **Faster R-CNN**
   - Architecture: Region Proposal Network (RPN)
   - Features: Region Proposal Network
   - Performance: 89% accuracy

5. **Inception-v3**
   - Architecture: Multi-scale processing
   - Features: Factorized convolutions, Auxiliary classifiers
   - Performance: 98% accuracy

6. **DenseNet121**
   - Architecture: Dense connectivity
   - Features: Fine-tuned for detailed classification
   - Performance: 96% accuracy

7. **XGBoost**
   - Type: Gradient boosting
   - Features: Feature importance, Handles imbalanced data
   - Performance: 98% accuracy

8. **Random Forest**
   - Type: Ensemble learning
   - Features: Feature selection, Parallel processing
   - Performance: 98% accuracy

### Technical Stack
- Python 3.8+
- PyTorch
- TensorFlow
- scikit-learn
- XGBoost
- OpenCV
- Streamlit

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU
- Git
- Docker (optional)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Download dataset
python scripts/preprocess.py --download
```

### Dataset Setup
```bash
# Process dataset
python scripts/preprocess.py --process
```

## Usage Guide

### Training Models
```bash
# Train specific model
python scripts/train.py --model resnet50 --epochs 100

# Train all models
python scripts/train.py --all
```

### Evaluation
```bash
# Evaluate model
python scripts/evaluate.py --model resnet50
```

### Web Interface
```bash
cd streamlit_app
streamlit run app.py
```

## Results & Performance

### Model Comparison

| Model           | Accuracy | Human Precision | Human Recall | Human F1 | Vehicle Precision | Vehicle Recall | Vehicle F1 |
|-----------------|----------|------------------|--------------|----------|--------------------|-----------------|------------|
| **ResNet50**        | 0.98     | 0.88             | 0.99         | 0.93     | 1.00               | 0.98            | 0.99       |
| **EfficientNetB0**  | 0.16     | 0.15             | 0.99         | 0.26     | 0.95               | 0.02            | 0.04       |
| **MobileNetV2**     | 0.99     | 0.95             | 0.98         | 0.96     | 1.00               | 0.99            | 0.99       |
| **DenseNet121**     | 0.96     | 0.77             | 0.99         | 0.87     | 1.00               | 0.95            | 0.97       |
| **InceptionV3**     | 0.98     | 0.92             | 0.93         | 0.92     | 0.99               | 0.99            | 0.99       |
| **Random Forest**   | 0.98     | 0.96             | 0.88         | 0.92     | 0.98               | 0.99            | 0.99       |
| **Faster R-CNN**    | 0.89     | 0.64             | 0.60         | 0.62     | 0.93               | 0.94            | 0.94       |
| **XGBoost**         | 0.98     | 0.96             | 0.94         | 0.95     | 0.99               | 0.99            | 0.99       |


### Key Findings
1. DenseNet-121 showed best performance for detailed classification
2. Deep learning models consistently outperformed traditional ML approaches
3. MobileNetV2 provided best speed-accuracy trade-off
4. Vision Transformer showed promising results but required more training data

## Future Work
1. Implement ensemble methods
2. Add real-time video processing
3. Optimize for edge deployment
4. Expand dataset with synthetic data
5. Implement cross-validation

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License & Contact

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact Information
- Bhushan Asati : [basati@stevens.edu]
- Rujuta Dabke : [rdabke@stevens.edu]
- Suyash Madhavi: [smadhavi1@stevens.edu]
- Anirudh Sharma : [asharma16@stevens.edu]

### Repository
Project Link: [https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git](https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git)
