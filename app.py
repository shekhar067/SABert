import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import string

app = Flask(__name__)

# Load the trained model using pickle
model = pickle.load(open('model.pkl', 'rb'))

# Clean text function to preprocess input text
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Home route to render HTML GUI
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict sentiment and render result on HTML GUI
@app.route('/getSentiment', methods=['POST'])
def getSentiment():
    '''
    For rendering results on HTML GUI
    '''
    # Extract input statement from form data
    statement = list(request.form.values())[0]
    # Clean the input statement
    cleaned_statement = clean_text(statement)
    # Get sentiment prediction
    output = get_result(cleaned_statement)

    return render_template('index.html', SentimentText='It\'s a {} statement'.format(output))

# Route for direct API calls
@app.route('/getsentiment_api', methods=['POST'])
def getsentiment_api():
    '''
    For direct API calls through request
    '''
    # Get JSON data
    data = request.get_json(force=True)
    # Extract input statement from JSON data
    statement = list(data.values())[0]
    # Clean the input statement
    cleaned_statement = clean_text(statement)
    # Get sentiment prediction
    sentiment_output = get_result(cleaned_statement)

    return jsonify(sentiment_output)

# Function to get sentiment prediction
def get_result(statement):
    result = model.predict([statement])
    pos = np.where(result[1][0] == np.amax(result[1][0]))
    pos = int(pos[0])
    sentiment_dict = {1: 'positive', 0: 'negative'}
    return sentiment_dict[pos]

if __name__ == "__main__":
    app.run(debug=True)
