import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

data_dir = os.path.join('.','data')
model_dir = os.path.join('.','models')


def load_data(test = False) :
    if test == False :
        train_data = pd.read_csv(os.path.join(data_dir,'wdbc train.csv'))	#pd.read_csv(train_path)
        labelencoder = LabelEncoder()
                    
        train_data['diagnosis'] = labelencoder.fit_transform(train_data['diagnosis'])

        y_train = train_data.iloc[:, 0].values
        x_train = train_data.iloc[:, 1:].values

        labelencoder_X_1 = LabelEncoder()
        y_train = labelencoder_X_1.fit_transform(y_train)

        print('Train Dataset \n',train_data.head(),'\n\n')

        return x_train, y_train

    elif test == True :
        test_data = pd.read_csv(os.path.join(data_dir,'wdbc test.csv'))	#pd.read_csv(test_path)
        labelencoder = LabelEncoder()
                
        test_data['diagnosis'] = labelencoder.fit_transform(test_data['diagnosis'])

        y_test = test_data.iloc[:, 0].values
        x_test = test_data.iloc[:, 1:].values

        labelencoder_X_1 = LabelEncoder()
        y_test = labelencoder_X_1.fit_transform(y_test)

        print('Test Dataset \n',test_data.head(),'\n\n')

        return x_test, y_test



def plot_history(history) : 
    fig_acc = plt.subplot(221)
    fig_acc.margins(0.05)  
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best', prop={'size': 6})

    fig_loss = plt.subplot(222)
    fig_loss.margins(0.05)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best', prop={'size': 6})
    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(cm) :
    heatmap_ticks = ['0', '1']
        
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(heatmap_ticks)))
    ax.set_yticks(np.arange(len(heatmap_ticks)))

    ax.set_xticklabels(heatmap_ticks)
    ax.set_yticklabels(heatmap_ticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(heatmap_ticks)):
        for j in range(len(heatmap_ticks)):
            text = ax.text(j, i, cm[i, j],
            ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    fig.tight_layout()
    plt.show(fig)

    
def save_model(model):
    model.save('saved_model')


def model_create() :
    model = Sequential()
    model.add(Dense(31, activation='relu', input_dim=30))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model



def model_train() :
    x_train, y_train = load_data(test = False)
    x_test, y_test = load_data(test = True)
    model = model_create()
            
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size=100, epochs=200, verbose=2)            

    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    
    print('Training Accuracy: %.2f%%' % (train_accuracy*100))

    save_model(model)

    plot_history(history)



def model_test() :
    model = keras.models.load_model('saved_model')
    x_test, y_test = load_data(test = True)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test Accuracy: %.2f%%' % (test_accuracy * 100))

    test_preds_classes = model.predict_classes(x_test)

    test_preds_df = pd.DataFrame(data = {'Actual' : y_test, 'Predicted' : test_preds_classes.flatten()})

    print('Predicted Lables on test data vs Actual Labels \n', test_preds_df)

    cm = confusion_matrix(y_test, test_preds_classes)

    plot_confusion_matrix(cm)


    
def model_predict() :
    model = keras.models.load_model('saved_model')

    query_columns = np.array(['meanradius', 'meantexture', 'meanperimeter', 'meanarea', 'meansmoothness', 'meancompactness', 'meanconcavity', 'meanconcavepoints',	'meansymmetry', 'meanfractaldimension', 'seradius', 
    'setexture','seperimeter', 'searea', 'sesmoothness', 'secompactness', 'seconcavity', 'seconcavepoints', 'sesymmetry', 'sefractaldimension', 'worstradius', 'worsttexture', 'worstperimeter', 'worstarea', 
    'worstsmoothness', 'worstcompactness', 'worstconcavity', 'worstconcavepoints', 'worstsymmetry', 'worstfractaldimension'])

        
    query_values = np.array([-0.201756035, 0, 0, -0.2714550596, 1.029197687, 0.8641183587, 0.7336389793, 0.8566968842, 1.120327751, 1.553584804, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1)


    query_df = pd.DataFrame(query_values, columns = query_columns)
        
    print(query_df)

    y_pred = model.predict_classes(query_df)

    print('Only 8 eight select features are used for making predictions at this moment using the values provided in the input boxes in the sidebar. The values for all the other features are set to 0')

    if y_pred == 0 :
        print('\n\n\n\nBreast Cancer Prediction : Benign')
        
    elif y_pred == 1 :
        print('\n\n\n\nBreast Cancer Prediction : Malign')
        


if __name__ == '__main__' :
    model_train()

    model_test()

    model_predict()
