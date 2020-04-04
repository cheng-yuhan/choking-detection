from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import os
import cv2
from data_loader import data_loader
from test_loader import test_loader
import matplotlib.pyplot as plt
import json
# generate dataset
train_choking_x,train_choking_y = data_loader.read("train_choking")
vali_choking_x, vali_choking_y = data_loader.read("validation_choking")
test_choking_x  = test_loader.read("test_choking")

train_choking_x = train_choking_x.astype("float32") /255
vali_choking_x = vali_choking_x.astype("float32") /255
test_choking_x = test_choking_x.astype("float32") / 255

num,time, w, h, color = train_choking_x.shape
model = models.Sequential()

model_recog = models.Sequential()
model_recog.add(layers.Reshape((time, -1)))
model_recog.add(layers.LSTM(units=64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model_recog.add(layers.LSTM(units=32, dropout=0.1, recurrent_dropout=0.1))
#model.add(layers.Dense(32, activation="relu"))
model_recog.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=optimizers.RMSprop(lr=2*1e-4),
              loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_choking_x, train_choking_y, epochs=1, batch_size=40,
                    validation_data=(vali_choking_x, vali_choking_y))
model.save("choking_model.h5")
y_pred = model.predict(test_choking_x)
#print("real_label", vali_choking_y)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Choking probability')
plt.legend()
plt.show()
#predict plt
time = range(1, len(y_pred) + 1)
plt.plot(time, y_pred)
plt.xlabel("time")
plt.ylabel("probability")
plt.title('choking prediction')
plt.legend()
plt.show()
#json file generate
jsontext = {"points": []}
for i in range(len(y_pred)):
  jsontext["points"].append({"Time":str(i), "predict_result":str(y_pred[i])})
jsondata = json.dumps(jsontext,indent=4,separators=(",", ": "))
f = open("/content/drive/My Drive/training/filename.json", "w")
f.write(jsondata)
f.close()
