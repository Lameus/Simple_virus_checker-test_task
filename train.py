import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib

def model(data):
    '''
    

    Function to create and fit model. Uses tf-idf for processing and logistic regretion
    for classification

    '''
    train_df = pd.read_csv(data, sep='\t') #download train dataset
    X_train, y_train = train_df.drop(columns=['is_virus', 'filename'], axis=1), train_df['is_virus']
    tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=1)
    logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs',
                          random_state=17, verbose=1)
    tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf),
                                ('logit', logit)])
    train = X_train['libs'].str.split(r'.dll\W*') #Remove dll from name of lib
    train_df1 = X_train
    train_df1['libs'] = train
    train_df1['libs'] = train_df1['libs'].str.join(' ')
    tfidf_logit_pipeline.fit(train_df1['libs'], y_train)
    
    joblib.dump(tfidf_logit_pipeline, "model.pkl") #Save the model
    
def main():
    # data = input('Enter data path: ')
    data = './data/train.tsv'
    model(data)

if __name__ == '__main__':
    main()