from keras.models import load_model
load_model = load_model("choking_model.h5")   # type in the filepath of the saved model
from test_loader import test_loader
import matplotlib.pyplot as plt

test_choking_x  = test_loader.read("test_choking")
test_choking_x = test_choking_x.astype("float32") / 255

y_pred = load_model.predict(test_choking_x)

time = range(1, len(y_pred) + 1)
plt.plot(time, y_pred)
plt.xlabel("time")
plt.ylabel("probability")
plt.title('choking prediction')
plt.legend()
plt.show()
