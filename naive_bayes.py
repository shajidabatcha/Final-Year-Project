import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("medical_data.csv")

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Convert text data to a matrix of token counts
vectorizer = CountVectorizer(stop_words='english')
train_matrix = vectorizer.fit_transform(train_data['text'])
test_matrix = vectorizer.transform(test_data['text'])

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(train_matrix, train_data['label'])

# Make predictions on the test set
y_pred = model.predict(test_matrix)

# Calculate the accuracy score
accuracy = accuracy_score(test_data['label'], y_pred)

# Create a word cloud of the most important words
word_freq = dict(zip(vectorizer.get_feature_names(), model.coef_[0]))
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Print the accuracy score
print("Accuracy:", accuracy)

