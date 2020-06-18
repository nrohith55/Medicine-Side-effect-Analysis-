from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix # import metrics
from sklearn.model_selection import cross_val_score # import evaluation tools
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


app = Flask(__name__)

#decorator (@app.route('/')) to specify the URL that should trigger the execution of the home function

@app.route('/')
def home():
	return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("train1.csv", encoding="latin-1")

	
	# Features and Labels
    
    
	df['label'] = df['output'].map({'No': 0, 'Yes': 1})
	X = df['review']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	

	clf = LinearSVC(C=0.5)
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)