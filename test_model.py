from keras.models import load_model
import keras.backend as K
#Define precision rate
def Prec(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

#Define Recall rate
def Rec(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall

load_model = load_model("choking_model_final.h5",     # type in the filepath of the saved model
custom_objects={'Prec': Prec,'Rec':Rec})
from test_loader import test_loader
import matplotlib.pyplot as plt
import numpy as np
import json

factor = 0.05
test_loader = test_loader()
test_choking_x  = test_loader.read("test_choking")  #type in the filepath of the test folder

y_pred = load_model.predict(test_choking_x)
y_pred = np.array(y_pred)-factor
time = range(1, len(y_pred) + 1)

jsontext = {"points": []}
for i in range(len(y_pred)):
  jsontext["points"].append({"Time":str(i), "predict_result":str(y_pred[i])})
jsondata = json.dumps(jsontext,indent=4,separators=(",", ": "))
f = open("data summary/video.json", "w")            # type in the filepath you want to save the json file
f.write(jsondata)
f.close()

plt.plot(time, y_pred)
plt.xlabel("time")
plt.ylabel("probability")
plt.title('choking prediction')
plt.legend()
plt.show()
