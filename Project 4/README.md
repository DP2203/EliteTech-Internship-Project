# Spam Email Detection - Machine Learning Model Implementation

A comprehensive machine learning project that implements multiple algorithms to classify emails as spam or legitimate (ham) using scikit-learn.

## 📋 Project Overview

This project demonstrates the complete machine learning workflow for text classification:
- **Data preprocessing** with TF-IDF vectorization
- **Multiple ML algorithms** (Logistic Regression, Random Forest, SVM)
- **Model evaluation** with cross-validation and multiple metrics
- **Interactive testing** system for new emails
- **Visualization** of results and feature importance

## 🎯 Learning Objectives

- Implement text preprocessing using TF-IDF vectorization
- Train multiple machine learning models using scikit-learn
- Evaluate model performance using various metrics
- Perform cross-validation for robust model assessment
- Visualize results and feature importance
- Create an interactive prediction system

## 📁 Project Structure

```
Project 4/
├── spam_detection_model.py        # Main Python script with complete implementation
├── spam.csv                      # Dataset with email text and labels
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Python Script

```bash
python spam_detection_model.py
```

- The script will load the data, train and evaluate models, display results and visualizations, and test example emails.
- All outputs will be printed to the console, and plots will be shown as pop-up windows.

## 📊 Dataset

**File**: `spam.csv`

**Format**:
```csv
text,label
"Hi John, how are you? Let's catch up this week.",0
"Congratulations! You've won a free iPhone. Claim it now!",1
```

**Description**:
- **10 emails** (5 spam, 5 ham)
- **Text column**: Email content
- **Label column**: 0 = Ham (legitimate), 1 = Spam

## 🤖 Machine Learning Models

The project implements and compares three algorithms:

1. **Logistic Regression**
   - Linear classification model
   - Good baseline performance
   - Fast training and prediction

2. **Random Forest**
   - Ensemble method using decision trees
   - Provides feature importance
   - Robust to overfitting

3. **Support Vector Machine (SVM)**
   - Effective for high-dimensional data
   - Good for text classification
   - Kernel-based learning

## 📈 Model Evaluation

### Metrics Used:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 3-fold CV for robust evaluation

### Visualizations:
- Confusion Matrix
- Model Performance Comparison
- Feature Importance (for Random Forest)
- Label Distribution

## 🎯 Interactive Features

### Email Testing Function

After running the script, you can use the `test_email()` function (defined inside the script) to classify new emails. To do this, you can:

- Modify the script to call `test_email()` with your own email text at the end of the script, or
- Copy the `test_email` function from the script into your own Python session after running the script, and use it with the trained model in memory.

**Note:** There is no saved model file. The model is only available during the script's execution.

### Example Emails Included:
- Legitimate emails (meeting reminders, work updates)
- Spam emails (lottery wins, free offers, urgent requests)
- Phishing attempts
- Business communications

## 🔧 Technical Implementation

### Text Preprocessing:
- **TF-IDF Vectorization**: Converts text to numerical features
- **Stop Words Removal**: Removes common English words
- **Max Features**: Limits to top 1000 features

### Model Pipeline:
```python
Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('classifier', LogisticRegression())
])
```

### Cross-Validation:
- **3-fold CV** for robust evaluation
- **Stratified sampling** to maintain class balance
- **Multiple metrics** for comprehensive assessment

## 🛠️ Requirements

### Python Version:
- Python 3.8 or higher

### Key Dependencies:
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations

## 🎓 Educational Value

This project demonstrates:
- **Complete ML workflow** from data to deployment
- **Multiple algorithm comparison**
- **Proper evaluation techniques**
- **Interactive testing systems**
- **Professional documentation**

Perfect for:
- Machine Learning courses
- Data Science projects
- Text classification tutorials
- Scikit-learn learning

## 🔮 Future Enhancements

Potential improvements:
- **Larger dataset** for better generalization
- **Deep learning models** (LSTM, BERT)
- **Additional features** (email headers, links, sender info)
- **Real-time email filtering**
- **Web interface** for user interaction
- **API deployment** for production use

## 📝 Usage Examples

### Running the Complete Analysis:
1. Run `python spam_detection_model.py`
2. View results and visualizations in your terminal and pop-up windows
3. After the classification report, check the message indicating if the last tested email is spam or not spam.
4. Test your own emails by modifying the script to call `test_email()` with your own text

## 🆕 Output Update

After running the script, you will see:
- A detailed classification report for the best-performing model.
- Immediately following the classification report, the script will automatically test the last example email and print a clear message:
  - `The tested email is spam.`  
  - or  
  - `The tested email is not spam.`

This message provides an immediate, human-readable summary of the model's prediction for the last example email in the list.

### Example Output

```
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.95        10
           1       1.00      0.80      0.89         5

    accuracy                           0.93        15
   macro avg       0.95      0.90      0.92        15
weighted avg       0.94      0.93      0.93        15

Email: You are the 1000th visitor! Click here to claim your reward.
Prediction: SPAM
It is spam.
Confidence: 99.99%
Spam Probability: 99.99%
Ham Probability: 0.01%
The tested email is spam.
```

## 🤝 Contributing

Feel free to enhance the project by:
- Adding new ML algorithms
- Improving visualizations
- Expanding the dataset
- Adding more features
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

---

**Created for**: Machine Learning Model Implementation Assignment  
**Topic**: Spam Email Detection using Scikit-learn  
**Deliverable**: Python Script with Complete Implementation and Evaluation 