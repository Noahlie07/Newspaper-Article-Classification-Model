import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


###############################################################################################################
## Part 1: Data Preprocessing

# Load in the dataset
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Dropping irrelevant columns
df = df.drop(columns=["link", "authors", "date"])

# Dropping duplicate rows and rows containing missing values
df = df.drop_duplicates()
df = df.dropna()

# Load in the necessary NLTK tools
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the stop words and Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df["short_description"] = df["short_description"].apply(text_preprocessing)
df["headline"] = df["headline"].apply(text_preprocessing)

#######################################################################################################
## Part 2: Feature Engineering

# Label Encoding for target variable
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(df[['short_description', 'headline']], df['category_encoded'], test_size=0.2, random_state=7)

# Converting Description and Headlines into numerical representations
vectorizer = CountVectorizer()
X_train_short_desc = vectorizer.fit_transform(X_train['short_description'])
X_test_short_desc = vectorizer.transform(X_test['short_description'])
X_train_headline = vectorizer.fit_transform(X_train['headline'])
X_test_headline = vectorizer.transform(X_test['headline'])


#######################################################################################################
## Part 3: Machine Learning

# Combining the feature matrices
X_train_combined = hstack([X_train_short_desc, X_train_headline])
X_test_combined = hstack([X_test_short_desc, X_test_headline])

# Logistic Regression model training
model = LogisticRegression(max_iter=1000, random_state=7)
model.fit(X_train_combined, y_train)

# Generating predictions
y_pred = model.predict(X_test_combined)

# Calculating Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Generating classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



