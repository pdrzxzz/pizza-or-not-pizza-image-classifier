# Pizza or Not Pizza - Image Classifier  

## 🛜 Available Here
https://pdrzxzz-pizza-or-not-pizza-image-classifier-app-izhml5.streamlit.app/

## 📌 Project Overview  

This project is an image classifier that determines whether an image contains pizza or not—**without using deep learning**.

This project provided comprehensive hands-on experience with traditional computer vision techniques, highlighting that while these methods can still yield results with proper feature engineering, deep learning approaches are significantly more powerful and accurate for image classification tasks.

### 🛠 Core Technologies  

- **Python**: Main language for the entire pipeline.  
- **Scikit-learn**: Handles ML models (Random Forest, SVM), PCA, and data preprocessing.  
- **Scikit-image**: Extracts features like LBP, HOG, and edges from images.  
- **Streamlit**: Powers the web app for image uploads and predictions.  
- **NumPy/Pandas**: Manages numerical operations and dataset handling.  
- **Matplotlib/Seaborn**: Visualizes learning curves and results.  
- **Joblib**: Saves and loads trained models for deployment.  
- **Imageio**: Reads and preprocesses uploaded images.  
- **Albumentations**: Augments training data with rotations and flips.

### Key Machine Learning Components
1. **Feature Extraction Techniques**:
   - Grayscale pixel values
   - Color histograms (RGB)
   - Local Binary Patterns (LBP) for texture
   - Sobel/Prewitt edge detection
   - Histogram of Oriented Gradients (HOG)

2. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) with 95% variance retention

3. **Classification Models**
   - Support Vector Machines (SVC)
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Decision Trees
   - Naive Bayes

4. **Model Evaluation**:
   - Stratified K-Fold Cross Validation
   - Learning Curve Analysis
   - Hyperparameter Tuning with GridSearchCV
   - Classification Reports (Precision, Recall, F1-score)

5. **Data Augmentation**:
   - Horizontal flips
   - Random brightness/contrast
   - Gaussian blur
   - Rotation (up to 15 degrees)
   
## 📚 Key Learnings

### 1. Traditional Computer Vision Challenges
- Learned that working with images without deep learning requires extensive feature engineering
- Discovered the importance of combining multiple feature extraction methods
- Understood the limitations of traditional approaches compared to CNN-based solutions

### 2. Feature Engineering
- Implemented various feature extraction techniques:
  - **Color Histograms**: Extracted and concatenated RGB channel histograms
  - **Texture Analysis**: Used Local Binary Patterns (LBP) with uniform patterns
  - **Edge Detection**: Applied Sobel and Prewitt operators
  - **HOG**: Extracted Histogram of Oriented Gradients features
- Learned to normalize and combine different feature types effectively

### 3. Dimensionality Reduction
- Gained practical experience with PCA for reducing feature space
- Learned to determine optimal number of components (95% variance retention)
- Understood the impact of PCA on model performance and training time

### 4. Model Development Process
- Implemented a complete ML pipeline from data loading to deployment
- Compared multiple classifiers and evaluated their performance
- Learned to interpret classification reports and confusion matrices
- Gained experience with stratified k-fold cross-validation

### 5. Hyperparameter Tuning
- Conducted systematic hyperparameter optimization using GridSearchCV
- Learned to define appropriate parameter grids for different models
- Understood the trade-offs between model complexity and performance

### 6. Data Augmentation
- Implemented image augmentation with Albumentations library
- Learned augmentation techniques to improve model generalization:
  - Random flips
  - Brightness/contrast adjustments
  - Small rotations
- Understood the importance of augmentation especially with limited data

### 7. Debugging and Development Practices
- Discovered the value of print statements for debugging complex pipelines
- Learned to monitor feature extraction shapes and data transformations
- Implemented progress tracking during lengthy operations

### 8. Deployment
- Created a Streamlit web application for model inference
- Learned to handle file uploads and display predictions
- Implemented proper model serialization with joblib

### 9. Visualization
- Generated learning curves to diagnose model behavior
- Created visualizations of predictions with actual vs predicted labels
- Implemented proper figure sizing and labeling for clarity


## 🏗️ Project Structure
```
pizza-not-pizza/
├── app.py                 # Streamlit application
├── feature_extractor.py   # Image feature extraction utilities
├── model_files/           # Serialized models and transformers
│   ├── model.pkl          # Trained classifier
│   ├── pca.pkl            # PCA transformer
│   └── scaler.pkl         # StandardScaler
├── notebooks/             # Jupyter notebook for model development
│   └── training.ipynb     # Complete training pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 How to Run  
1. Install python
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Launch the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
   if this command don't work, try
   ```bash
   python -m streamlit run app.py
   ```  
4. Upload an image via the UI to test!  
---  
*Note: The dataset (`pizza-not-pizza`) was sourced from Kaggle using `kagglehub`.*
