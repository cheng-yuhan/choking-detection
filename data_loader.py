import os
import cv2
import numpy as np

class data_loader():
    def read(data_dir):
        video_data = []
        video_label = []
        for file in os.listdir(data_dir):
            clip_1 = []
            clip_2 = []
            name = file.split(sep='_')
            file_dir = os.path.join(data_dir, file)
            cap = cv2.VideoCapture(file_dir)
            fps = cap.get(cv2.CAP_PROP_FPS)
            #print("current pics per second", int(fps))
            times =  0
            while True:
                times += 1
                res, image = cap.read()
                if not res:
                    #print('not res , not image')
                    break
                w, h, color = image.shape
                image = cv2.resize(image, (128, 72))
                clip_1.append(image)
                if times > 30:
                    clip_2.append(image)
                if times % 60 == 0:
                    video_data.append(np.array(clip_1))
                    if times > 30 * int(name[1]):
                        video_label.append(1)
                    else:
                        video_label.append(0)
                    clip_1 = []
                elif times % 60 == 30 and times > 30:
                    video_data.append(np.array(clip_2))
                    clip_2 = []
                    if times > 30 * int(name[1]):
                        video_label.append(1)
                    else:
                        video_label.append(0)
        video_data = np.array(video_data)
        video_label = np.array(video_label)
        print(video_data.shape)
        print(video_label.shape)
        return video_data,video_label



if __name__ == '__main__':
    data_loader.read("train_choking")