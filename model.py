import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,precision_score,f1_score, recall_score, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Download stopwords
nltk.download('stopwords')

#dataset
data = pd.read_csv('feedback.csv')

# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

data['Processed_Feedback'] = data['Feedback'].apply(preprocess_text)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data['Processed_Feedback'], data['Emotion'], test_size=0.2, random_state=42)

# vectorization and classification
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')

# Calculate F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1:.2f}')

recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall:.2f}')

y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Save the model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
