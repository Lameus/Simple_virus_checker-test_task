import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, recall_score 
from sklearn.metrics import f1_score

import joblib

def stats(data):
    '''
    
    Takes path of validation data, gives model's stats 

    '''
    tfidf_logit_pipeline = joblib.load("model.pkl") #dowload the model
    val_df = pd.read_csv(data, sep='\t')
    X_val, y_val = val_df.drop(columns=['is_virus', 'filename'], axis=1), val_df['is_virus']
    val = X_val['libs'].str.split(r'.dll\W*')
    X_val['libs'] = val
    X_val['libs'] = X_val['libs'].str.join(' ')
    valid_pred = tfidf_logit_pipeline.predict(X_val['libs'])
    #scoring
    tn, fp, fn, tp = confusion_matrix(y_val, valid_pred).ravel()
    aps = average_precision_score(y_val, valid_pred)
    rs = recall_score(y_val, valid_pred)
    acs = accuracy_score(y_val, valid_pred)
    f1 = f1_score(y_val, valid_pred)
    
    #Writing
    file = open("./results/validation.txt", "w")
    file.write(f'True positive: {tp} \n')
    file.write(f'False positive: {fp} \n')
    file.write(f'False negative: {fn} \n')
    file.write(f'True negative: {tn} \n')
    file.write(f'Accuracy: {acs} \n')
    file.write(f'Precision: {aps} \n')
    file.write(f'Recall: {rs} \n')
    file.write(f'F1: {f1}')
    file.close()
    
def main():
    # data = input('Enter data path: ')
    data = './data/val.tsv'
    stats(data)

if __name__ == '__main__':
    main()