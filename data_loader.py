import os
import cv2
import numpy as np
from keras import layers
from keras import models
from keras.applications import VGG16,InceptionV3


class data_loader():
    def feature_model(self):
        model = models.Sequential()
        conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(216, 144, 3))
        conv_base.trainable = False
        model.add(conv_base)
        return model

    def read(self, data_dir):
        model = self.feature_model()
        video_data = []
        video_label = []
        for file in os.listdir(data_dir):
            clip_1 = []
            clip_2 = []
            name = file.split(sep='_')
            file_dir = os.path.join(data_dir, file)
            cap = cv2.VideoCapture(file_dir)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("current pics per second", int(fps))
            times =  0
            while True:
                times += 1
                res, image = cap.read()
                if not res:
                    #print('not res , not image')
                    break
                #print(image.shape)
                image = cv2.resize(image, (36, 64))
                image = (image - image.min(axis=0)) / (image.max(axis=0) - image.min(axis=0))

                clip_1.append(image)
                if times > 30:
                    clip_2.append(image)
                if times % 60 == 0:
                    clip_1 = model.predict(np.array(clip_1))
                    video_data.append(np.array(clip_1))
                    if times > 30 * int(name[1]):
                        video_label.append(1)
                    else:
                        video_label.append(0)
                    clip_1 = []
                elif times % 60 == 30 and times > 30:
                    clip_2 = model.predict(np.array(clip_2))
                    video_data.append(np.array(clip_2))
                    clip_2 = []
                    if times > 30 * int(name[1]):
                        video_label.append(1)
                    else:
                        video_label.append(0)
        video_data = np.array(video_data)
        video_label = np.array(video_label)
        print(video_label.shape)
        return video_data,video_label



if __name__ == '__main__':
    data_loader = data_loader()
    data_loader.read("train_choking")
