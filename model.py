# Import necessary libraries for data processing and analysis
import numpy as np  # For numerical calculations
import pandas as pd  # For data processing using DataFrames
import os  # For working with file paths and directories
import re  # For regular expressions and text cleaning
import string  # For working with string-related operations
from sklearn.model_selection import train_test_split  # For data splitting
from simpletransformers.classification import ClassificationModel  # For building and training the model
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For creating plots
import sklearn.metrics  # For model evaluation metrics
import pickle  # For model serialization
import sklearn
############## DATA PreProcessing ##############

# Iterate through all text files in the specified directory and print their paths
#for dirname, _, filenames in os.walk('InputDS'):
#    for filename in filenames:
#       print(os.path.join(dirname, filename))

# Import data from text files using pandas
df1 = pd.read_csv('InputDS/ds1.txt', delimiter='\t', header=None)
df2 = pd.read_csv('InputDS/ds2.txt', delimiter='\t', header=None)
df3 = pd.read_csv('InputDS/ds3.txt', delimiter='\t', header=None)

# Merge the three datasets into a single DataFrame
df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# Add column headers to the DataFrame
df.columns = ['Review', 'Score']


# Display the shape of the DataFrame
df.shape

# Get information about the dataset
df.info()

# Find missing values in the dataset
df.isnull().sum()

########### EDA ############

# Display the top 5 records to verify columns and data format
df.head()

# Display the bottom 5 records to verify columns and data format
df.tail()

# Provide more information about the numerical column of the dataset
df.describe()

# Check the distribution of positive and negative cases
df.Score.value_counts()


 #Positve and negative cases using Seaborn Plot - to graphically respsent the blance of dataset

sns.countplot(data=df,x='Score')
plt.xlabel('Score');



#To clean the text and remove the numbers and links

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# call the clean method to clean the dataset
df['Review'] = df['Review'].apply(lambda x:clean_text(x))

#check the data after cleaning
df.head()

#spliting train test dataset


#train 80% and test data 20%
train_df,test_df = train_test_split(df,test_size = 0.2)





#create a model from Bert pretrained
model = ClassificationModel('bert', 'bert-base-cased', num_labels=2, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

#train the model with training data
model.train_model(train_df)

#get result of the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)

#print result
result

#print model output
model_outputs

#get model output to a list
lst = []
for arr in model_outputs:
    lst.append(np.argmax(arr))

#get test score and will check test predeiction    
test_score = test_df['Score'].tolist()
predicted = lst


mat = sklearn.metrics.confusion_matrix(test_score , predicted)

#confusion matrix
mat

#prsent in heat map
df_cm = pd.DataFrame(mat, range(2), range(2))

sns.heatmap(df_cm, annot=True) 
plt.show()

#get classification resport
sklearn.metrics.classification_report(test_score,predicted,target_names=['positive','negative'])

#Check model accuracy
sklearn.metrics.accuracy_score(test_score,predicted)
# Define a function to get sentiment prediction for a statement
def get_result(statement):
    # Load the trained model
    #loaded_model = ClassificationModel('bert', 'bert-base-cased', num_labels=2, args={'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda=False)
    
    # Preprocess the statement
    cleaned_statement = clean_text(statement)
    
    # Predict the sentiment
    result, model_outputs = model.predict([cleaned_statement])
    
    # Determine the sentiment label
    sentiment_label = np.argmax(model_outputs[0])
    sentiment_dict = {1: 'positive', 0: 'negative'}
    predicted_sentiment = sentiment_dict[sentiment_label]
    
    return predicted_sentiment

# sample inout
input_statement = "I really enjoyed the movie, it was amazing!"
predicted_sentiment = get_result(input_statement)
print(f"Predicted Sentiment: {predicted_sentiment}")


# Saving model to disk using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Loading model from disk using pickle
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

