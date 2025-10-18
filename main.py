import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import joblib
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
lem = WordNetLemmatizer()
stem = PorterStemmer()
def clean_text(text):
    training_data['Description'] = training_data['Description'].str.lower()
    training_data['Description'] = training_data['Description'].replace(r'[^\w\s]', '', regex=True)
    training_data['Description'] = training_data['Description'].replace(r'(<br\s/?>|br\s/?|/br)', '', regex=True)
    training_data['Description'] = training_data['Description'].apply( lambda x: " ".join([word for word in x.split() if word not in stop_words]))

    return training_data
training_data=pd.read_csv("train.csv")
testing_data=pd.read_csv("test.csv")
stop_words = set(stopwords.words('english'))
clean_training_data=clean_text(training_data['Description'])
clean_testing_data=clean_text(testing_data['Description'])

xtrain=clean_training_data['Description']
ytrain=clean_training_data['Class Index']

xtest=clean_testing_data['Description']
ytest=clean_testing_data['Class Index']

vectorizer= TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(xtrain)
x_test_vec = vectorizer.transform(xtest)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, ytrain)
y_pred = model.predict(x_test_vec)
print("Accuracy: ", accuracy_score(ytest,y_pred))
print(classification_report(ytest,y_pred))
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')

