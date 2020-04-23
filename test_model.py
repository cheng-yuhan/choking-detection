from keras.models import load_model
load_model = load_model("choking_model.h5")   # type in the filepath of the saved model
from test_loader import test_loader
import matplotlib.pyplot as plt
import json

test_loader = test_loader()
test_choking_x  = test_loader.read("test_choking")

y_pred = load_model.predict(test_choking_x)

jsontext = {"points": []}
for i in range(len(y_pred)):
  jsontext["points"].append({"Time":str(i), "predict_result":str(y_pred[i])})
jsondata = json.dumps(jsontext,indent=4,separators=(",", ": "))
f = open("filename.json", "w")       # type in the filepath you want to save the json file
f.write(jsondata)
f.close()


time = range(1, len(y_pred) + 1)
plt.plot(time, y_pred)
plt.xlabel("time")
plt.ylabel("probability")
plt.title('choking prediction')
plt.legend()
plt.show()
