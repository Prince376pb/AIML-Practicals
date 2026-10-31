# Practical 07 - Ensemble Learning Techniques

## Practical Title
Implementation and Evaluation of Ensemble Learning Techniques

## Problem Statement
Implement Stacking, AdaBoost, and Voting ensemble techniques. Compare their performance with individual machine learning models using classification metrics.

## Dataset Used
Iris Flower Dataset

- Total Records: 150
- Total Features: 4
- Classes: Setosa, Versicolor, Virginica
- Training Data: 80%
- Testing Data: 20%

## Algorithms Used

**Base Models:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

**Ensemble Models:**
- Stacking Classifier
- AdaBoost Classifier
- Voting Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Technologies Used
- Python
- Google Colab
- Pandas
- Matplotlib
- Scikit-learn

## Results

The performance of all models was compared using the testing dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| KNN | 0.9333 | 0.9444 | 0.9333 | 0.9327 |
| Stacking | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| AdaBoost | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Voting | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

**Best Model:** Stacking Classifier with 96.67% accuracy.

## Output
The program displays:
- Iris dataset details
- Classification reports
- Model performance comparison table
- Performance comparison graph
- Best-performing model

## Conclusion
In this practical, I implemented Stacking, AdaBoost, and Voting ensemble techniques using the Iris Flower dataset. I compared their performance with individual machine learning models using accuracy, precision, recall, and F1-score.

The Stacking Classifier achieved the highest accuracy of 96.67% in this experiment. Through this practical, I learned how ensemble techniques combine multiple models to make predictions.
