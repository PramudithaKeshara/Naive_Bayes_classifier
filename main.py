from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data - you should replace this with your labeled dataset
good_words = ['positive', 'excellent', 'awesome', 'fantastic']
bad_words = ['negative', 'horrible', 'awful', 'terrible']

# Creating labeled data
X = good_words + bad_words
y = [1] * len(good_words) + [0] * len(bad_words)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the words using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Making predictions on the test set
predictions = classifier.predict(X_test_vec)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Using the model to predict new words
new_words = ['amazing', 'terrible', 'great', 'horrible']
new_words_vec = vectorizer.transform(new_words)
predictions_new = classifier.predict(new_words_vec)

# Displaying the predictions for new words
for word, prediction in zip(new_words, predictions_new):
    label = 'Good' if prediction == 1 else 'Bad'
    print(f'{word}: {label}')
