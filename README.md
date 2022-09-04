# Discriminant Analysis

# Mr. John Hughes is looking at developing an LDA model for his cancer.csv dataset and evaluate its effectiveness. If you recall the dataset has the following variables.

# Independent Variables
ID - ID number
Clump Thickness - 1-10
UofCSize - Uniformity of Cell Size 1-10 UofShape - Uniformity of Cell Shape 1-10 Marginal Adhesion - 1-10
SECSize - Single Epithelial Cell Size 1-10 Bare Nuclei - 1-10
Bland Chromatin - 1-10
Normal Nucleoli - 1-10
Mitoses - 1-10


# Dependent Variable
Class
Benign (i.e. No Cancer) 2
Malignant (i.e. Cancer) - 4

Note: ID will not be used and will need to be dropped prior to building your model

Below are the results of the Optimized Logistical Regression model (with SMOTE):
      Optimized Model
      Model Name: LogisticRegression(class_weight='balanced', random_state
      =100)
      Best Parameters: {'clf__C': 1, 'clf__penalty': 'l2'}
1
    [[89  0]
 [ 3 45]]
    accuracy
   macro avg
weighted avg
precision    recall  f1-score   support
2 4
0.97      1.00
1.00      0.94
0.98      0.97
0.98      0.98
0.98        89
0.97        48
0.98       137
0.98       137
0.98       137

# The Ask:
1. Rational Statement (summary of the problem or problems to be addressed by the PPT) 
2. Identify and explain two (2) key insights from the Pandas Profile Report 
3. Present and explain three (3) key insights from the Optimized LDA classification report, but first use
4. SMOTE to ensure that the dataset is balanced
5. Compare the Optimized LDA to the Optimized Logistical Regression model (from page 1) identifying
three (3) key insights
6. State and explain two (2) recommendations for Mr. John Hughes for next steps
