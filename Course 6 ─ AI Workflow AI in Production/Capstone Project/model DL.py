import pandas as pd
import numpy as np
import os
import time
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


def load_data() :
    # Get the path of the directory containing the data
    data_dir = os.path.join('.','data')

    # Read the data into pandas DataFrame
    train_df = pd.read_csv(os.path.join(data_dir,'wdbc train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir,'wdbc test.csv'))
    
    # OneHotLabelEncode the Label Values
    #train_df['diagnosis'] = train_df['diagnosis'].replace({'B': '0', 'M': '1'})
    #test_df ['diagnosis'] = test_df['diagnosis'].replace({'B': '0', 'M': '1'})

    # Get the Features and the Label
    Y_train = train_df['diagnosis']
    X_train = train_df.iloc[:,1:]
    #print(X_train)

    Y_test = test_df['diagnosis']
    X_test = test_df.iloc[:, 1:]

    return X_train, Y_train, X_test, Y_test

def model_train(test = False) :
    ## Start timer for runtime
    time_start = time.time()

    # Ingest Data
    X_train, Y_train, X_test, Y_test = load_data()

    # Create the model
    model = Sequential()
    model.add(Dense(31, activation='relu', input_dim=30))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size=100, epochs=200)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("Accuracy: %.2f%%" % (scores[1]*100))

        

    '''if test:
        print("... saving test version of model")
        pickle.dump(grid_rf,os.path.join("models","test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        pickle.dump(grid_rf,SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models",'latest-train.pickle')
        with open(data_file,'wb') as tmp:
            pickle.dump({'y':y,'X':X},tmp)
        
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_train_log(X.shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=test)'''
    
model_train()
