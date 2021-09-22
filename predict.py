import pandas as pd
import numpy as np
import eli5
import joblib

def prediction(data):
    '''
    
    Takes path of test data, classifies virus status 

    '''
    tfidf_logit_pipeline = joblib.load("model.pkl") #download the model
    test_df = pd.read_csv(data, sep='\t')
    test = test_df['libs'].str.split(r'.dll\W*')
    test_df['libs'] = test
    test_df['libs'] = test_df['libs'].str.join(' ')
    test_df['Prediction'] = tfidf_logit_pipeline.predict(test_df['libs']) #classification
    np.savetxt('./results/prediction.txt', test_df['Prediction'], fmt='%d', header='prediction',
              comments='') 
    return test_df, tfidf_logit_pipeline

def reasons(data, pipeline):
    '''
    
    Creates txt file with explanation

    '''
    info = []
    for i in range(data.shape[0]):
        if data.loc[i, 'Prediction'] == 1:
            #creates explanation in special format
            a = eli5.explain_prediction_sklearn(pipeline.named_steps['logit'],
                                doc=data['libs'].iloc[i], vec=pipeline.named_steps['tf_idf'])
            b = eli5.formatters.text.format_as_text(a, show_feature_values=False,
                                       show=('targets', 'feature_importances'))
            bb = b.split('\n')
            if len(bb) > 8:
                info.append(bb[3:8]) #appends only top 5 reasons
            else:
                info.append(bb[3:len(bb) - 1]) #appends all reasons
        else:
            info.append(' ')
    #strip spaces in elements of "info"
    fixed_info = [[info[i][j].strip() for j in range(len(info[i]))] for i in range(len(info))]
    data['explain'] = fixed_info #connection between explanation and ids
    tfile = open('./results/explain.txt', 'w')
    tfile.write('The main libs of "virus" marked files and weights of them \n')
    data['explain'] = data['explain'].str.join(' | ')
    tfile.write(data['explain'].to_string())
    tfile.close()
    
def main():
    # data = input('Enter data path: ')
    data = './data/test.tsv'
    preds, pipeline = prediction(data)
    reasons(preds, pipeline)
    
if __name__ == '__main__':
    main()