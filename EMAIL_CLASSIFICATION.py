# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 2. Load and clean
df = pd.read_csv('student_email_dataset.csv')
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Email_Text'])

# 3. Label distribution plot
sns.countplot(x='Label', data=df, palette='Set2')
plt.title("Email Category Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='dotted')
plt.show()

# 4. Feature extraction
X_raw = df['Email_Text']
y = df['Label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_raw)

# 5. Split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Confusion matrix
labels = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title("Confusion Matrix")
plt.show()

# 7. Predicting new sample correctly
sample = ["harsh are you stupid"]
# 🔑 Use the same fitted vectorizer — not a new one
sample_vec = vectorizer.transform(sample)
print("Prediction for sample email:", model.predict(sample_vec)[0])