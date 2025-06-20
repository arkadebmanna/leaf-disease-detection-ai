# Leaf Disease Detection AI

A deep learning-based project for detecting diseases in crop leaves using image classification. This application enables users—such as farmers, agronomists, and agricultural researchers—to upload leaf images and receive real-time predictions on the type of disease, using a trained CNN model.

## Project Summary

- Detects multiple leaf diseases across various crops.
- Built using Convolutional Neural Networks with transfer learning (e.g., MobileNetV2).
- Provides Grad-CAM heatmaps to visualize model attention.
- Optional: Deployable on Microsoft Azure using Azure Web App or Azure Machine Learning.

## Folder Structure

```
leaf-disease-detection-ai/
├── README.md  
├── LICENSE  
├── .gitignore  
├── data/  
│   ├── sample_images/  
│   └── PlantVillage_metadata.csv  
├── notebooks/  
│   └── multicrop_leaf_disease.ipynb  
├── src/  
│   ├── data_preprocessing.py  
│   ├── model_training.py  
│   ├── predict.py  
│   └── utils.py  
├── model/  
│   └── leaf_model.h5  
├── app/  
│   ├── app.py  
│   ├── templates/index.html  
│   └── static/styles.css  
├── requirements.txt  
└── screenshots/  

```

## How to Run Locally

1. Clone the Repository:


2. Create Virtual Environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
.
env\Scripts ctivate         # Windows
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask App:

```bash
cd app
python app.py
```


## Deployment

Option 1: Local Deployment  
Use Python and Flask as shown above.

Option 2: Azure Deployment  
- Upload model/leaf_model.h5 to Azure Blob Storage or attach to Azure ML Workspace.  
- Deploy the app folder via Azure Web App or Azure App Service.  
- For model inference, use Azure Functions or deploy via Azure Machine Learning Service.

## Model Evaluation

- Accuracy: ~93%
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Visual Aids:  
  - Confusion Matrix  
  - Accuracy vs Loss Curves  
  - Grad-CAM Heatmaps  
  - Prediction vs Actual Chart

## Dataset

- PlantVillage Dataset (public)
- Contains labeled images of healthy and diseased leaves for various crops

## References

- Mohanty, Sharada P., et al. "Using Deep Learning for Image-Based Plant Disease Detection." Frontiers in Plant Science, 2016.
- TensorFlow Documentation
- PlantVillage Dataset on GitHub and Kaggle

## License

MIT License – see the LICENSE file for details.
