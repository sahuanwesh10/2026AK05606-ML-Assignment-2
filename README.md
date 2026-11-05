# 2026AK05606-ML-Assignment-2
# Dry Bean Classification: 5 ML Models + Streamlit App

**Name:** Anwesh Sahu  **BITS ID:** 2026AK05606

## a. Problem statement
The goal is to predict the variety of a dry bean from measurements of its shape and size. There are 7 varieties: Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira. This is a **multi-class classification** problem. Sorting beans by hand is slow and inconsistent, so an automatic classifier based on measurements taken from images could help standardise quality checks.

Five classification models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same dataset. They are compared using six metrics: Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The trained models are deployed in an interactive Streamlit web app.

## b. Dataset description
- **Name:** Dry Bean Dataset
- **Source:** Kaggle (https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset)
- **Instances:** 13,611
- **Features:** 16 (all numeric)
- **Target column:** `Class` (7 categories)
- **Missing values:** none
- **Class counts in the full dataset:** Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522 (the classes are imbalanced)

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4. These are shape and size measurements extracted from bean images.

**Preprocessing**
- Rows with missing values are dropped (none were found).
- The text class labels are label-encoded to numbers.
- Features are standardised with `StandardScaler`, which is included inside every model pipeline. The scaler is therefore saved with each model and applied automatically in the app.
- The data is split 80% train and 20% test, stratified by class, with `random_state=42`. The 20% test part is saved as `test_data.csv`.

**Metric settings (multi-class):** AUC uses one-vs-rest with macro averaging. Precision, Recall and F1 use weighted averaging, which accounts for the class imbalance.

## c. GitHub repository link
https://github.com/sahuanwesh10/2026AK05606-ML-Assignment-2

**Repository contents**

```
├── app.py                          (Streamlit app)
├── requirements.txt
├── README.md
├── test_data.csv                   (20% hold-out test data)
├── ML_Assignment2_Training.ipynb   (training and evaluation code)
└── model/                          (5 trained models + meta.joblib)
```

## d. Models used

All 5 models are trained on the same training split. Each is a scikit-learn Pipeline of `StandardScaler` followed by the classifier.

| Model | Settings |
|---|---|
| Logistic Regression | max_iter=2000 |
| Decision Tree | max_depth=10 |
| KNN | n_neighbors=7 |
| Naive Bayes | GaussianNB |
| Support Vector Machine | RBF kernel, probability=True |

### Comparison table
| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9214 | 0.9948 |      0.9222 |   0.9214 | 0.9216 | 0.905  |
| Decision Tree          |     0.9071 | 0.968  |      0.9074 |   0.9071 | 0.9072 | 0.8876 |
| KNN                    |     0.9137 | 0.9858 |      0.9144 |   0.9137 | 0.9139 | 0.8956 |
| Naive Bayes            |     0.8979 | 0.9916 |      0.9007 |   0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine |     0.9221 | 0.9949 |      0.9224 |   0.9221 | 0.9222 | 0.9059 |

Best model by MCC: Support Vector Machine
Best model by Accuracy: Support Vector Machine


### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed very strongly, with Accuracy 0.9214, AUC 0.9948 and MCC 0.905, almost the same as the best model. This suggests the 7 bean classes are largely separable by a linear boundary once the features are standardised. It is also fast to train and easy to interpret. |
| Decision Tree | Accuracy 0.9071 and MCC 0.8876 are good, but it has the lowest AUC (0.968) of all five models. A single tree gives coarse class probabilities, so it ranks classes less reliably than the smoother models. |
| KNN | Accuracy 0.9137, F1 0.9139 and MCC 0.8956 put it in the middle of the group, ahead of the Decision Tree and Naive Bayes but behind Logistic Regression and SVM. AUC 0.9858 is good but lower than Logistic Regression and SVM. It depends on feature scaling, which the pipeline handles. |
| Naive Bayes | Lowest Accuracy (0.8979), F1 (0.8981) and MCC (0.8773), though still above 0.87 on MCC. Its AUC is high (0.9916), so its probabilities rank the classes well even when its final predictions are less accurate. A likely reason is that Naive Bayes assumes features are independent, while bean features such as Area, Perimeter and ConvexArea are strongly correlated. |
| Support Vector Machine | Best model on every metric: Accuracy 0.9221, AUC 0.9949, Precision 0.9224, Recall 0.9221, F1 0.9222 and MCC 0.9059. The RBF kernel can learn curved class boundaries. Its lead over Logistic Regression is very small (about 0.0007 in Accuracy). |
| **Overall winner for your dataset?** | **Support Vector Machine**, which ranks first by Accuracy, AUC, F1 and MCC. Logistic Regression is a very close second and is much simpler and faster, so it is a good practical alternative. |

## Streamlit app features

- Upload a test CSV (it must contain all 16 feature columns and the `Class` column)
- Dropdown to select any of the 5 models
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- A tab comparing all 5 models on the uploaded data
- Predictions table with a CSV download

The app loads the saved models and does not retrain on the uploaded data.

## How to run locally

```
pip install -r requirements.txt
streamlit run app.py
```
