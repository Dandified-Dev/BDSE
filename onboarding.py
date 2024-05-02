import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder  # Added import
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sqlalchemy import create_engine

# Database connection
engine = create_engine('mysql+pymysql://root:DandiVojuke74!@localhost/onboarding')

# Load and preprocess training data from the database
query = "SELECT * FROM train WHERE root_genre = 'Jazz'"
df = pd.read_sql(query, engine)
df.drop_duplicates(subset=df.columns, keep='first', inplace=True)
df['reviewText'] = df['reviewText'].str.lower()
df['overall'] = pd.cut(df['overall'], bins=[0, 2, 3, 5], labels=['negative', 'neutral', 'positive'], include_lowest=True)
df['reviewText'].fillna('', inplace=True)

# Define stop words and vectorize the text data
my_stop_words = list(ENGLISH_STOP_WORDS.union(['song', 'music', 'melody', 'lyrics', 'chorus', 'verse', 'artist', 'album', "band", "track", "tune", "sound", "beat", "rhythm", "instrument", "vocal", "guitar", "drum", "bass", "piano", "keyboard", "singer", "singing", "sing", "play", "listen", "listening", "cd", 'albums', 'songs', 'track', 'tracks', 'record', 'records', 'release', 'releases', 'released']))
vect = CountVectorizer(stop_words=my_stop_words)
vect.fit(df.reviewText)
X = vect.transform(df.reviewText)

# Encode target labels
y = df['overall']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# SVM Classifier
svm_classifier = SVC(kernel='linear', C=0.4, class_weight='balanced')
svm_classifier.fit(X_train, y_train)
predictions_svm = svm_classifier.predict(X_test)

# Evaluate SVM performance
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")
print("SVM Classification Report:\n", classification_report(y_test, predictions_svm))
conf_mat_svm = confusion_matrix(y_test, predictions_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_svm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix')

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)
predictions_rf = rf_classifier.predict(X_test)

# Evaluate Random Forest performance
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Random Forest Classification Report:\n", classification_report(y_test, predictions_rf))
conf_mat_rf = confusion_matrix(y_test, predictions_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')

# Unseen Data Predictions
query = "SELECT * FROM train WHERE root_genre = 'Jazz'"
unseen_data =  pd.read_sql(query, engine)
unseen_text_data = unseen_data['reviewText'].str.lower()
X_unseen = vect.transform(unseen_text_data)

# SVM Predictions on Unseen Data
predictions_svm_unseen = svm_classifier.predict(X_unseen)
predictions_svm_unseen_labels = label_encoder.inverse_transform(predictions_svm_unseen)
print("SVM Predictions on Unseen Data:", predictions_svm_unseen_labels)

# Random Forest Predictions on Unseen Data
predictions_rf_unseen = rf_classifier.predict(X_unseen)
predictions_rf_unseen_labels = label_encoder.inverse_transform(predictions_rf_unseen)
print("Random Forest Predictions on Unseen Data:", predictions_rf_unseen_labels)

# Visualizations
plt.figure(figsize=(8, 6))
sns.countplot(predictions_svm_unseen_labels, palette="Blues", order=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('SVM Predicted Label Distribution on Unseen Data', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(predictions_rf_unseen_labels, palette="Blues", order=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Random Forest Predicted Label Distribution on Unseen Data', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()

# Visualize the 100 most used words with WordCloud
word_frequencies = dict(zip(vect.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate_from_frequencies(word_frequencies)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 100 Most Used Words', fontsize=16)
plt.show()
