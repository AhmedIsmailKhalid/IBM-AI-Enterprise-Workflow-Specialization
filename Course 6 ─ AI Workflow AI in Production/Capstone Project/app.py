import streamlit as st
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


class MyCallback(keras.callbacks.Callback):
    #def __init__(self, num_epochs):
    #    self._num_epochs = num_epochs
    def on_train_begin(self, logs=None):
        st.subheader('Training in Progress')
        st.markdown('Percentage Complete')
        self._progress = st.empty()
        self._epoch_header = st.empty()
        self._epoch_progress_header = st.empty()
    def on_train_end(self, logs=None):
        st.markdown('Training Done!')
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        for i in range(0, epoch) :
            self._progress.progress(i/198)
        self._epoch_header.text(f'Epoch {epoch+1}/{200}')
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self._epoch_progress_header.text(f'loss: {np.round(logs["loss"],6)}  accuracy: {np.round(logs["accuracy"],6)} \t val_loss: {np.round(logs["val_loss"],6)} \t val_accuracy: {np.round(logs["val_accuracy"],6)}')



def main() :
    st.title('Deep Learning Model on Streamlit Website')
    st.markdown('Testing Deep Learning on Streamlit Website')

    st.sidebar.header('Choose Optimizer and Loss Function to Train the Neural Network')
    st.sidebar.text('')
    optimizer = st.sidebar.selectbox('Optimizer', ('rmsprop', 'sgd', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl'))
    loss = st.sidebar.selectbox('Loss Function', ('binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson'))
    st.sidebar.text('')

    train_button = st.sidebar.button('Train Model', key = 'train model')
    test_button = st.sidebar.button('Test Model', key = 'test model')
    
    st.sidebar.subheader('Enter Values for Features to use for prediction')
    meanradius = st.sidebar.number_input('Mean Radius', key = 'mean radius')
    meanarea = st.sidebar.number_input('Mean Area', key = 'mean area')
    meansmoothness = st.sidebar.number_input('Mean Smoothness', key = 'mean smoothness')
    meancompactness = st.sidebar.number_input('Mean Compactness', key = 'mean compactness')
    meanconcavity = st.sidebar.number_input('Mean Concavity', key = 'mean concavity')
    meanconcavepoints = st.sidebar.number_input('Mean Concavepoints', key = 'mean concavepoints')
    meansymmetry = st.sidebar.number_input('Mean Symmetry', key = 'mean symmetry')
    meanfractaldimension = st.sidebar.number_input('Mean Fractal Dimension', key = 'mean fractal dimension')

    predict_button = st.sidebar.button('Make Prediction', key = 'predict model')


    data_dir = os.path.join('.','data')
    model_dir = os.path.join('.','models')

    # unless function or arguments change, cache outputs to disk and use cached outputs anytime app re-runs
    @st.cache(persist=True, suppress_st_warning=True)
    def load_data(test = False) :
        if test == False :
            #train_path = r'C:\Users\Ahmed Ismail Khalid\Desktop\Capstone Project\data\wdbc train.csv'
            train_data = pd.read_csv(os.path.join(data_dir,'wdbc train.csv'))	#pd.read_csv(train_path)
            labelencoder = LabelEncoder()
                
            train_data['diagnosis'] = labelencoder.fit_transform(train_data['diagnosis'])

            y_train = train_data.iloc[:, 0].values
            x_train = train_data.iloc[:, 1:].values

            
            st.dataframe(train_data)

            labelencoder_X_1 = LabelEncoder()
            y_train = labelencoder_X_1.fit_transform(y_train)

            return x_train, y_train

        elif test == True :
            #test_path = r'C:\Users\Ahmed Ismail Khalid\Desktop\Capstone Project\data\wdbc test.csv'
            test_data = pd.read_csv(os.path.join(data_dir,'wdbc test.csv'))	#pd.read_csv(test_path)
            labelencoder = LabelEncoder()
                
            test_data['diagnosis'] = labelencoder.fit_transform(test_data['diagnosis'])

            y_test = test_data.iloc[:, 0].values
            x_test = test_data.iloc[:, 1:].values

            labelencoder_X_1 = LabelEncoder()
            y_test = labelencoder_X_1.fit_transform(y_test)

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

        st.pyplot()

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
        plt.setp(ax.get_xticklabels(), ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(heatmap_ticks)):
            for j in range(len(heatmap_ticks)):
                text = ax.text(j, i, cm[i, j],
                            ha="center", va="center", color="w")

        ax.set_title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        fig.tight_layout()
        st.pyplot(fig)

    
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

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        st.text(f'Optimizer: {optimizer}     Loss Function: {loss}')

        return model



    def model_train() :
        x_train, y_train = load_data(test = False)
        x_test, y_test = load_data(test = True)
        model = model_create()

        mycustomcallback = MyCallback()

            
        history = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size=100, epochs=200,callbacks=[mycustomcallback], verbose=2)            

        train_loss, train_accuracy = model.evaluate(x_train, y_train)
        st.write('')
        st.write('Training Accuracy: %.2f%%' % (train_accuracy*100))

        save_model(model)

        plot_history(history)

    
    def model_test() :
        st.markdown('This returns the models accuracy on the test data. The saved model is loaded and is provided the x_test and y_test to evaluate its performance. To make predictions using trained/saved first train the model (if you want to train a new model with different optimizer and loss function) and then use the Predict Button')
        model = keras.models.load_model('saved_model')
        x_test, y_test = load_data(test = True)

        st.write('')
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        st.write('')
        st.write('Test Accuracy: %.2f%%' % (test_accuracy * 100))
        st.markdown('Testing Evaluation Finished!')

        test_preds_classes = model.predict_classes(x_test)

        test_preds_df = pd.DataFrame(data = {'Actual' : y_test, 'Predicted' : test_preds_classes.flatten()})

        st.subheader('Predicted Lables on test data vs Actual Labels')
        st.write(test_preds_df)

        st.write('')
        
        cm = confusion_matrix(y_test, test_preds_classes)

        plot_confusion_matrix(cm)

    def model_predict() :
        model = keras.models.load_model('saved_model')

        query_columns = np.array(['meanradius', 'meantexture', 'meanperimeter', 'meanarea', 'meansmoothness', 'meancompactness', 'meanconcavity', 'meanconcavepoints',	'meansymmetry', 'meanfractaldimension', 'seradius', 
        'setexture','seperimeter', 'searea', 'sesmoothness', 'secompactness', 'seconcavity', 'seconcavepoints', 'sesymmetry', 'sefractaldimension', 'worstradius', 'worsttexture', 'worstperimeter', 'worstarea', 
        'worstsmoothness', 'worstcompactness', 'worstconcavity', 'worstconcavepoints', 'worstsymmetry', 'worstfractaldimension'])

        
        query_values = np.array([meanradius, 0, 0, meanarea, meansmoothness, meancompactness, meanconcavity, meanconcavepoints, meansymmetry, meanfractaldimension, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,-1)


        query_df = pd.DataFrame(query_values, columns = query_columns)
        
        st.write(query_df)

        y_pred = model.predict_classes(query_df)

        st.markdown('Only 8 eight select features are used for making predictions at this moment using the values provided in the input boxes in the sidebar. The values for all the other features are set to 0')

        st.subheader('Breast Cancer Prediction')
        if y_pred == 0 :
            st.write('Benign')
        elif y_pred == 1 :
            st.write('Malign')
        

    if train_button :
        model_train()

    if test_button :
        model_test()

    if predict_button :
        model_predict()

if __name__ == '__main__' :
    main()