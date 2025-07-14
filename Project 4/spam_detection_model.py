import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
import joblib
import sys

warnings.filterwarnings('ignore')

def main():
    # 1. Load the Dataset
    try:
        df = pd.read_csv('spam.csv')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found. Please make sure the file is in the correct directory.")
        sys.exit(1)

    # 2. Explore the Data
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # Convert labels to numerical values (0 and 1) if not already
    if df['label'].dtype != int:
        df['label'] = df['label'].map({'ham': 0, 'spam': 1, '0': 0, '1': 1}).astype(int)

    # 3. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # 4. Create Multiple Models for Comparison
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
            ('classifier', LogisticRegression(max_iter=200, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
            ('classifier', SVC(probability=True, random_state=42))
        ])
    }

    # 5. Train and Evaluate All Models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)

    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        print(f"{name} Cross-validation scores: {cv_scores}")
        print(f"{name} CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nBest Model: {best_accuracy:.4f} accuracy")

    # 6. Detailed Analysis of Best Model
    print("\n" + "="*50)
    print("DETAILED ANALYSIS OF BEST MODEL")
    print("="*50)

    y_pred_best = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    # 7. Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 8. Feature Importance Analysis (for Random Forest)
    if 'Random Forest' in models and best_model == models['Random Forest']:
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
        feature_importance = best_model.named_steps['classifier'].feature_importances_
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_df.head(10))
        plt.figure(figsize=(10, 6))
        top_features = feature_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # 9. Interactive Email Testing Function
    def test_email(email_text):
        prediction = best_model.predict([email_text])[0]
        probability = best_model.predict_proba([email_text])[0]
        print(f"\nEmail: {email_text}")
        print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
        if prediction == 1:
            print("It is spam.")
        else:
            print("It is not spam.")
        print(f"Confidence: {probability[prediction]:.2%}")
        print(f"Spam Probability: {probability[1]:.2%}")
        print(f"Ham Probability: {probability[0]:.2%}")
        return prediction, probability

    # 10. Test Some Example Emails
    print("\n" + "="*50)
    print("INTERACTIVE EMAIL TESTING")
    print("="*50)

    test_emails = [
        "Hi John, how are you? Let's catch up this week.",
        "Congratulations! You've won a free iPhone. Claim it now!",
        "Meeting reminder: Tomorrow at 10 AM in the conference room.",
        "URGENT! You have won a lottery prize of $100000. Contact us immediately.",
        "Thanks for the update. I'll review the documents and get back to you.",
        "Limited time offer: Get 50% off on all items. Shop now!",
        "Can you please send me the report by end of day?",
        "Your bank account has been compromised. Click here to secure it.",
        "Reminder: Doctor's appointment on Friday at 2 PM.",
        "You are the 1000th visitor! Click here to claim your reward."
    ]

    last_prediction = None
    for email in test_emails:
        last_prediction, _ = test_email(email)
        print("-" * 30)

    # Print message about the last tested email
    if last_prediction == 1:
        print("The tested email is spam.")
    else:
        print("The tested email is not spam.")

    # 11. Model Performance Summary
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Best Model Accuracy: {best_accuracy:.4f}")
    print(f"Dataset Size: {len(df)} emails")
    print(f"Training Set: {len(X_train)} emails")
    print(f"Testing Set: {len(X_test)} emails")
    print(f"Spam Ratio: {(df['label'] == 1).mean():.2%}")

    # 12. Save the Best Model
    # joblib.dump(best_model, 'spam_detector_model.pkl')
    # print(f"\nBest model saved as 'spam_detector_model.pkl'")

    print("\n" + "="*50)
    print("SPAM DETECTION SYSTEM READY!")
    print("="*50)
    print("You can now use the test_email() function to classify new emails.")
    print("Example: test_email('Your email text here')")

if __name__ == "__main__":
    main() 