# Heart Disease Prediction
Introduction:
The Cardiovascular Heart Disease Prediction project uses a dataset of 1025 samples and 14 features, some of which include age, gender, chest pain type, resting blood pressure, serum cholestoral, fasting blood sugar, maximum heart rate achieved etc .Data analysis has been done to arrive at the number of patients. The machine learning model helps us in determining whether .The goal of the project is to provide a detailed analysis and accurate affected with heart disease and what are the major attributes contributing for the same. The machine learning model helps us in determining whether the patient is believed to have exposed to heart disease or not.The goal of the project is to provide a detailed analysis and accurate prediction from the real world dataset available on kaggle.

Dataset:
Heart Disease Dataset
Source: Kaggle
https://www.kaggle.com/datasets/johnsmith8
8/heart-disease-dataset?resource=download

MODEL SELECTION :
We've selected Random Forestmodel for cardiovascular heartdisease prediction.Among all the models on which wehave trained and tested our data,we've finalised this model due to itsexceptional performance (model accuracy), versatility, and scalability.
Random Forest model is performing exceptionally well on this dataset, with high accuracy of 1, precision, recall, & F1-Score. The hyperparameter tuning process led to apply the best hyperparameters for the Random Forest classifier.

TRAINING MODEL WITH REGULARIZATION :
We have used Regularization techniques to prevent overfitting in Random forest. It adds a penalty term to the model's cost function to discourage it from fitting the noise in the training data and to promote a simpler and more generalizable model. The results also suggests that L1 regularization is more effective in reducing overfitting and achieving better accuracy than L2 regularization.

RESULTS & CONCLUSIONS
We conducted a predictive analysis using a Random Forest classifier to predict the presence of heartdisease based on various cardiovascular risk factors. To prevent overfitting, we employed L1 and L2 regularization techniques. The model achieved an accuracy of 96.11% on the test set and demonstrated high precision, recall, and F1-score for both classes.
We also performed hyperparameter tuning using cross-validation and obtained the best hyperparameters (max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=50).
Data augmentation was utilized to increase the dataset's diversity and enhance the model's generalizationability.
The Random Forest classifier showed robust performance in correctly classifying individuals with and without heart disease, making it a reliable tool for this classification task.
