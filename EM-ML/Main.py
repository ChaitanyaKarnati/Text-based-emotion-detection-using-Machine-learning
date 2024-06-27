
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from emoji import UNICODE_EMOJI
import os
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from string import punctuation
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.models import model_from_json

stop_words = set(stopwords.words('english'))
os.environ["PYTHONIOENCODING"] = "utf-8"


main = tkinter.Tk()
main.title("Emotion Detection using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, Y_train, Y_test
global tokenizer
global XX
global model
global positive_score,negative_score,neutral_score

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def checkInput(inputdata):
    option = 0
    try:
        s = float(inputdata)
        option = 0
    except:
        option = 1
    return option

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    return tokens

def Preprocessing():
    global X
    global Y
    X = []
    Y = []
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding = "ISO-8859-1")
    count = 0
    for i in range(len(train)):
        #ids = train.get_value(i,0,takeable = True)
        sentiment = train.get_value(i,0,takeable = True)
        tweet = train.get_value(i,1,takeable = True)
        check = checkInput(tweet)
        if check == 1:
            tweet = tweet.lower().strip()
            tweet = clean_doc(tweet)
            print(str(i)+" == "+str(sentiment)+" "+tweet)
            '''
            icon = train.get_value(i,2,takeable = True)
            if str(icon) != 'nan':
                icon = UNICODE_EMOJI[icon]
                icon = ''.join(re.sub('[^A-Za-z\s]+', '', icon))
                icon = icon.lower()
            else:
                icon = ''
            '''    
            arr = tweet.split(" ")
            msg = ''
            for k in range(len(arr)):
                word = arr[k].strip()
                if len(word) > 2 and word not in stop_words:
                    msg+=word+" "
            textdata = msg.strip()  #+" "+icon
            X.append(textdata)
            Y.append((sentiment-1))
            count = count + len(arr)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y = np.nan_to_num(Y)
    print(Y)
    #Y = np.asarray(Y)#pd.get_dummies(train['sentiment']).values
    Y = to_categorical(Y)
    print(Y)
    text.insert(END,'Total reviews found in dataset : '+str(len(X))+"\n")
    text.insert(END,'Total words found in all reviews : '+str(count)+"\n")

def generateModel():
    text.delete('1.0', END)
    global XX
    global tokenizer
    global X_train, X_test, Y_train, Y_test
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X)
    XX = tokenizer.texts_to_sequences(X)
    XX = pad_sequences(XX)
    X_train, X_test, Y_train, Y_test = train_test_split(XX,Y, test_size = 0.13, random_state = 42)
    text.insert(END,'Total features extracted from reviews are  : '+str(X_train.shape[1])+"\n")
    text.insert(END,'Total splitted records used for training : '+str(len(X_train))+"\n")
    text.insert(END,'Total splitted records used for testing : '+str(len(X_test))+"\n") 

def buildClassifier():
    global model
    global XX
    global Y
    text.delete('1.0', END)
    embed_dim = 70
    lstm_out = 70
    max_fatures = 500

    indices = np.arange(XX.shape[0])
    np.random.shuffle(indices)

    XX = XX[indices]
    Y = Y[indices]
    
    print(XX.shape)
    print(Y)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()
        text.insert(END,'Dataset trained uisng LSTM Algorithm. See black console for LSTM layers details\n')
    else:
        model = Sequential()
        model.add(Embedding(max_fatures, embed_dim,input_length = XX.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        batch_size = 256
        model.fit(XX, Y, epochs = 100, batch_size=batch_size, verbose = 1)
        model.save_weights('model/model_weights.h5')            
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        text.insert(END,'Dataset trained uisng LSTM Algorithm. See black console for LSTM layers details\n')
    print(model.summary())        

def module1():
    global positive_score,negative_score,neutral_score
    text.delete('1.0', END)
    sentence = tf1.get()
    sentence = sentence.lower().strip()
    sentence = clean_doc(sentence)
    textdata = sentence.strip()
    mytext = [textdata]
    twts = tokenizer.texts_to_sequences(mytext)
    twts = pad_sequences(twts, maxlen=433, dtype='int32', value=0)
    sentiment = model.predict(twts,batch_size=1,verbose = 2)[0]
    print(sentiment)
    result = np.argmax(sentiment)
    if result == 0:
        text.insert(END,sentence+' is classified as very NEGATIVE\n\n')
    if result == 1:
        text.insert(END,sentence+' is classified as NEGATIVE\n\n')
    if result == 2:
        text.insert(END,sentence+' is classified as NEUTRAL\n\n')
    if result == 3:
        text.insert(END,sentence+' is classified as POSITIVE\n\n')
    if result == 4:
        text.insert(END,sentence+' is classified as Very Positive\n\n')    
    
    height = sentiment
    bars = ('Very Negative', 'Negative','Neutral','Positive','Very Positive')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()    
    
    

font = ('times', 16, 'bold')
title = Label(main, text='Emotion Detection using Machine Learning')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Emotion Dataset To Train Model", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Preprocessing", command=Preprocessing)
preButton.place(x=370,y=100)
preButton.config(font=font1) 

modelButton = Button(main, text="Generate Train Test Model", command=generateModel)
modelButton.place(x=510,y=100)
modelButton.config(font=font1) 

classifierButton = Button(main, text="Build Classifier", command=buildClassifier)
classifierButton.place(x=730,y=100)
classifierButton.config(font=font1) 

l1 = Label(main, text='Enter A Comment:')
l1.config(font=font1)
l1.place(x=50,y=150)

tf1 = Entry(main,width=40)
tf1.config(font=font1)
tf1.place(x=200,y=150)

runButton = Button(main, text="Run", command=module1)
runButton.place(x=550,y=150)
runButton.config(font=font1)

#main.config(bg='OliveDrab2')
main.mainloop()
