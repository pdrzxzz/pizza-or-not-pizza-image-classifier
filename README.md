# Pizza or Not Pizza - Image Classifier  

## 📌 Project Overview  

This project is an image classifier that determines whether an image contains pizza or not—**without using deep learning**. The traditional computer vision approach was chosen to explore fundamental image processing and machine learning techniques.  

## 🛠️ Technologies Used  

### Core Libraries  
- **Scikit-learn**: Machine learning models (Random Forest, SVM, etc.), PCA, and data preprocessing (`StandardScaler`, `LabelEncoder`)  
- **Scikit-image**: Image processing and feature extraction (LBP, HOG, Sobel, color histograms)  
- **Streamlit**: Web application for user interaction and image uploads  
- **Joblib**: Saving and loading trained models (`pca.pkl`, `scaler.pkl`, `model.pkl`)  
- **Imageio**: Image reading and preprocessing (`imread`, resizing, grayscale conversion)  

### Feature Extraction Techniques  
- **Grayscale Pixels**: Raw pixel values from resized grayscale images  
- **Color Histograms**: 256-bin histograms for each RGB channel (768 features total)  
- **Local Binary Patterns (LBP)**: Texture analysis with uniform patterns (`radius=1`, `n_points=8`)  
- **Edge Detection**: Sobel and Prewitt filters for gradient-based features  
- **HOG (Histogram of Oriented Gradients)**: Captures shape information (`pixels_per_cell=(8,8)`)  

### Data Augmentation (Training Phase)  
- **Albumentations**: Applied to diversify the dataset:  
  - Horizontal flips (`p=0.5`)  
  - Random brightness/contrast adjustments (`p=0.2`)  
  - Gaussian blur (`p=0.2`)  
  - Rotation (±15 degrees, `p=0.3`)  

### Model Training Pipeline  
1. **Data Splitting**: Stratified 80/20 train-test split to handle class imbalance.  
2. **Feature Standardization**: `StandardScaler` for normalization.  
3. **Dimensionality Reduction**: PCA (`n_components=0.95` variance retention).  
4. **Model Selection**: Evaluated multiple classifiers:  
   - Random Forest (best performance)  
   - SVM, K-Nearest Neighbors, Logistic Regression, Decision Trees, Naive Bayes  
5. **Hyperparameter Tuning**: GridSearchCV for optimizing `RandomForestClassifier`.  

### Deployment  
- **Streamlit App**: Simple UI to upload images and view predictions.  
- **Model Artifacts**: Pre-trained `PCA`, `Scaler`, and `Random Forest` models loaded for inference.  

## 📖 Key Learnings  
1. **Traditional vs. Deep Learning**:  
   - Manual feature engineering (e.g., HOG, LBP) is complex but educational.  
   - Deep learning (e.g., CNNs) often outperforms but requires more data/compute.  

2. **Debugging with Prints**:  
   - Extremely necessary while coding complex things being a effective like a bug catcher and a debbuging tool.

3. **PCA in Practice**:  
   - Reduced ~3K features to 95% variance with 50 components, speeding up inference.  

4. **Data Augmentation**:  
   - Improved model robustness by synthetically expanding the dataset.  

5. **Model Interpretability**:  
   - Random Forest provided transparency (vs. "black-box" neural networks).  

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

## 🔍 Future Improvements  
- Experiment with **additional features** (e.g., SIFT, SURF).  
- Deploy as a **cloud-based API** (e.g., FastAPI + Docker).  
- **Quantify uncertainty** in predictions (e.g., probability scores).  

---  
*Note: The dataset (`pizza-not-pizza`) was sourced from Kaggle using `kagglehub`.*
