import emoji as emoj
import webbrowser
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter import filedialog
from keras.models import load_model
from flask import Flask, render_template , request
ch=""
app=Flask(__name__)
@app.route('/')
def Hand_Input():
   return render_template('Hand_Input.html')
@app.route('/Hand_Display', methods=['POST'])
def Hand_Display():
   ch=request.form['Choice']
   if(ch=='1'):
      mnist = tf.keras.datasets.mnist
      (x_train, y_train),(x_test,y_test)=mnist.load_data()
      x_train = tf.keras.utils.normalize(x_train , axis = 1)
      x_test = tf.keras.utils.normalize(x_test , axis = 1)
      model= tf.keras.models.Sequential()
      model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
      model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
      model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
      model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
      model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
      model.fit(x_train,y_train, epochs=3)
      loss , accuracy  =model.evaluate(x_test,y_test)
      print(accuracy)
      print(loss)
      def openFile():
        filepath = filedialog.askopenfilename(initialdir=r"C:\Users\ARNAB MONDAL\PycharmProjects\1st SEMESTER Python\img",
                                          title="Open file okay?",filetypes=(("text files","*.txt"),
                                          ("all files",".png")))
        global prediction
        Img=cv2.imread(filepath)[:,:,0]
        Img=np.invert(np.array([Img]))
        predict=(model.predict(Img))
        prediction=str(np.argmax(predict))
        plt.imshow(Img[0],cmap=plt.cm.binary)#change the color in black and white
        plt.show()
      while(True):
        window = Tk()
        button = Button(text="Open",command=openFile)
        button.pack()
        window.mainloop()
        break
   if(ch=='2'):
    model = load_model('handwritten_character_recog_model.h5')

    words = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
    def openFile():
       filepath = filedialog.askopenfilename(initialdir=r"C:\Users\ARNAB MONDAL\PycharmProjects\1st SEMESTER Python\img",
                                          title="Open file okay?",filetypes=(("text files","*.txt"),
                                          ("all files",".png")))
       global prediction
       Img=cv2.imread(filepath)[:,:,0]
       Img=np.invert(np.array([Img]))
       image = cv2.imread(filepath)
       image_copy= cv2.imread(filepath)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       image = cv2.resize(image, (400,440))
       image_copy = cv2.GaussianBlur(image_copy, (7,7), 0)
       gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
       _, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
       final_image = cv2.resize(img_thresh, (28,28))
       final_image =np.reshape(final_image, (1,28,28,1))

       prediction = words[np.argmax(model.predict(final_image))]
       print("----------------")
       print("The predicted value is : ",prediction)
       print("----------------")
       plt.imshow(Img[0],cmap=plt.cm.binary)#change the color in black and white
       plt.show()
    while(True):
     window = Tk()
     button = Button(text="Open",command=openFile)
     button.pack()
     window.mainloop()
     break
   return render_template('Hand_Display.html', Predict=prediction , show=emoj.emojize(":smiling_face_with_open_hands:") )
if __name__ == '__main__':
  webbrowser.open_new('http://127.0.0.1:5000')
  app.run()
