#!/usr/bin/env python
# coding: utf-8

# ## 範例
# 參考 `train.py` 實現的訓練模型程式碼範例
# 

# In[1]:



import tensorflow.compat.v1 as tf

print(tf.__version__)


import os,sys


# In[3]:


# In[4]:


import os, wget
if not os.path.exists("model_data/yolo.h5"):
  # 下載 yolov3 的網路權重，並且把權重轉換為 keras 能夠讀取的格式
  print("Model doesn't exist, downloading...")
  os.system("wget https://pjreddie.com/media/files/yolov3.weights")
  wget.download("https://pjreddie.com/media/files/yolov3.weights")
  print("Converting yolov3.weights to yolo.h5...")
  os.system("python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5")
else:
  print("Model exist")


# In[5]:



# 下載測試資料集

import urllib.request
if not os.path.isdir('test'):
        os.makedirs('test')
        urllib.request.urlretrieve('https://dehayf5mhw1h7.cloudfront.net/wp-content/uploads/sites/726/2017/08/03065555/Kangacoon-1024x768.jpg', 'test/Kangacoon1.jpg')

# 下載 raccoon 和 kangaroo 測試影片

if not os.path.isdir('video'):
        os.makedirs('video')
        urllib.request.urlretrieve('https://cvdl-fileentity.cupoy.com/2nd/homework/example/1580979702432/Raccoon.mp4', 'video/Raccoon.mp4')
        urllib.request.urlretrieve('https://cvdl-fileentity.cupoy.com/2nd/homework/example/1580979702432/Kangaroo.mp4', 'video/Kangaroo.mp4')



# In[6]:


import os
import numpy as np
import shutil

images_path = ['./raccoon_dataset/images/', './kangaroo/images/']  



for i, img in enumerate(images_path):  
    image_ids = os.listdir(img)  
    assert isinstance(image_ids, list), 

    image_ids.sort()  # 檔案名稱由小排到大
    image_ids_size = len(image_ids)  # 該 images 路徑下有多少檔案
    
    split_num = int(np.round(image_ids_size*0.97))  # 每個資料集的後 3% 當測試集
    test_data = image_ids[split_num:]  # test 拿後 3% 筆資料
    
    for j in range(len(test_data)):
        shutil.move(img+test_data[j], 'test') # 會在當前路徑中(keras-yolo3)建立一個 test 的資料夾
    print('Move done!')
    
    print('test_data from', img, ':', len(test_data))
    print(test_data)



if not os.path.exists("animals_train.txt"):
    import xml.etree.ElementTree as ET  # 載入能夠 Parser xml 文件的 library
    from os import getcwd
    
    sets = ['train', 'val']  # 分為訓練集和驗證集
    classes = ["raccoon", "kangaroo"]  # raccoon(第0類) 和 kangaroo(第1類) 的資料類別
    annots_path = ['./raccoon_dataset/annotations/', './kangaroo/annots/']  # annotation 路徑
    images_path = ['./raccoon_dataset/images/', './kangaroo/images/']  # image 路徑，其實上面建立test資料夾時就設過了
    
    # 把 annotation 轉換訓練時需要的資料形態
    def convert_annotation(annots_path, image_id, list_file):
        in_file = open('%s%s.xml'%(annots_path, image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
                
            cls_id = classes.index(cls)  # cls 分類至 cls_id(0 或 1)
            xmlbox = obj.find('bndbox')  # xmlbox 顯示影像方框(bounding box)
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), 
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            
    for image_set in sets:  # 跑 train 和 val
        annotation_path = 'animals_%s.txt'%(image_set)
        list_file = open(annotation_path, 'w')  # 開檔+寫檔
        print("save annotation at %s" % annotation_path)
        
        for i, annots in enumerate(annots_path):  # i 跑 0 和 1，annots 跑 raccoon 和 kangaroo 的 annotation 路徑
            #print(len([name for name in os.listdir(annots) if os.path.isfile(os.path.join(annots, name))]))
            image_ids = os.listdir(annots)  # 該 annotation 路徑下的檔案
            image_ids.sort()  # 檔案名稱由小排到大
            image_ids_size = len(image_ids)  # 該 annotation 路徑下有多少檔案
            split_num = int(np.round(image_ids_size*0.8))  # 每個資料集的前 80% 當訓練集，後 10% 當驗證集
            
            if image_set == 'train':
                data = image_ids[:split_num]  # train 拿前 80% 筆資料
            else:
                data = image_ids[split_num:]  # val 拿後 20% 筆資料
                
            for image_id in data:  # 跑 xml 的 id
                xml_name = image_id.split('.')[0]  # 用'.'切割字串，然後取 .xml 前面的名稱
                list_file.write('%s%s.jpg'%(images_path[i], xml_name))  # train 和 val 各別 跑 0 和 1 的 image 路徑
                convert_annotation(annots, xml_name, list_file)  # 呼叫上面定義的函式
                list_file.write('\n')
                
        list_file.close()  # 關檔(與開檔要對應位置)

filename = 'animals_train_aug.txt'
file = open(filename)
print(filename, 'has', len(file.readlines()), 'data，which are 80% of total')

with open(filename, 'r') as f:
  d = f.readlines()
#d
len(d)

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper

# 把 YOLO weights 轉換為能夠提供給 keras 作為訓練新模型的初始權重
if not os.path.exists("model_data/yolo_weights.h5"):
    print("Converting pretrained YOLOv3 weights for training")
    # '-w': 代表只轉換 yolov3.weights 到 model_data/yolo_weights.h5
    os.system("python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5")
    print("Done!")
else:
    print("Pretrained weights exists")
    
# 避免OSError: image file is truncated 故加上下方兩行程式碼
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 設定訓練資料的來源位置
annotation_path = 'animals_train_aug.txt'  # 轉換好格式的 train 標註檔案
log_dir = 'logs\\000\\'  # 訓練好的模型要儲存的路徑
classes_path = 'model_data/animals_classes.txt' #此txt檔，須先建立好第一行為raccoon第二行為kangaroo
anchors_path = 'model_data/yolo_anchors.txt'

class_names = get_classes(classes_path)
num_classes = len(class_names)
#num_classes = len(["kangaroo","raccoon"])
anchors = get_anchors(anchors_path)

# 模型參數的設定，並設定 logging, checkpoint, reduce_lr, early_stopping
input_shape = (416, 416)  # multiple of 32, hw
is_tiny_version = len(anchors)==6 # default setting

if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='model_data/yolo_weights.h5')

#logging = TensorBoard(log_dir=log_dir)
logging = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir)

checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', 
                             monitor='val_loss', 
                             save_weights_only=True, 
                             save_best_only=True, 
                             period=3)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=10,  # 3
                              verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', 
                               min_delta=0.001,  # 0
                               patience=20,  # 10
                               verbose=1)

# 分為 training 以及 validation
val_split = 0.25  # train:val = 80%:20% = 4:1

with open(annotation_path) as f:
    lines = f.readlines()
    
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)

num_val = int(len(lines)*val_split)
print(num_val)
num_train = len(lines) - num_val
print(num_train)
# 第一階段訓練
# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
# 一開始先 freeze YOLO 除了 output layer 以外的 darknet53 backbone 來 train
if True:
    model.compile(optimizer=Adam(lr=1e-3),loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # use custom yolo_loss Lambda layer
    print('\n第一階段訓練')
    
    batch_size = 16
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 模型利用 generator 產生的資料做訓練，強烈建議大家去閱讀及理解 data_generator_wrapper 在 train.py 中的實現
    model_1 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes), 
                                  steps_per_epoch=max(1, num_train//batch_size), 
                                  validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes), 
                                  validation_steps=max(1, num_val//batch_size), epochs=50, initial_epoch=0, callbacks=[logging, checkpoint])
    
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')

# 第二階段訓練
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
if True:
    # 把所有 layer 都改為 trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
    print('\n第二階段訓練: Unfreeze all of the layers.')

    batch_size = 8   # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    model_2 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes), 
                                  steps_per_epoch=max(1, num_train//batch_size), 
                                  validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes), 
                                  validation_steps=max(1, num_val//batch_size), epochs=100, initial_epoch=50, callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    
    model.save_weights(log_dir + 'trained_weights_final.h5')
    
# 分析模型
# 畫出訓練過程中 train_loss 與 val_loss 的變化
import matplotlib.pyplot as plt
#%matplotlib inline

# 第一階段訓練
plt.title('Learning rate - stage 1') # 對應至訓練中 unfreeze 前的階段 i.e. 前50次 epoch
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(model_1.history["loss"], label="train_loss")
plt.plot(model_1.history["val_loss"], label="val_loss")
plt.legend(loc='upper right')
plt.show()

# 第二階段訓練
plt.title('Learning rate - stage 2') # 對應至訓練中 unfreeze 後的階段 i.e. 後50次 epoch
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(model_2.history["loss"], label="train_loss")
plt.plot(model_2.history["val_loss"], label="val_loss")
plt.legend(loc='upper right')
plt.show()


'''
測試使用model:
'''
trained_model = log_dir + 'trained_weights_final.h5'

# 測試影像
from yolo import YOLO
classes_path = 'model_data/animals_classes.txt'
yolo_model = YOLO(model_path= trained_model, classes_path=classes_path)

from PIL import Image
from IPython import display

for img_file in os.listdir('test'):
    image = Image.open(os.path.join('test/', img_file))  # 讀取範例圖片
    r_image = yolo_model.detect_image(image)  # 執行 yolo 檢測，將回傳的圖片儲存在 r_image 中
    display.display(r_image)  # 顯示 r_image，可觀察到圖片上已畫上 yolov3 所檢測的 object
    
    
# 偵測影片
from yolo import YOLO
import numpy as np
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

log_dir = 'logs\\000\\' # 訓練好的模型要儲存的路徑
classes_path = 'model_data/animals_classes.txt'

yolo_model = YOLO(model_path= trained_model, classes_path=classes_path)

def detect_video(yolo, video_path, output_path=""):
    # 透過 OpenCV 擷取影像
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video!")
        
    # 取得影像的基本資訊
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*'MP4V')  # 指定 video 編碼方式(mp4)
    video_fps = vid.get(cv2.CAP_PROP_FPS)  # 總共有多少 frames
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),   # 每個 frame 的寬
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 每個 frame 的高
    
    # 設定影像的輸出
    isOutput = True if output_path != "" else False
    if isOutput:
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    
    # 初始化設定
    video_cnt = 0  
    video_playtime = []  
    
    # 迭代每個 frame 來進行影像偵測
    while True:
        return_value, frame = vid.read() # 讀取每個 frame
        video_cnt += 1 
        
        # 先把每個 frame 分開偵測，再把偵測完的 frames 串接回影片，最後輸出偵測好的影片
        if return_value == True: 
            image = Image.fromarray(frame)
            start_time = time.time() 
            image = yolo.detect_image(image)  # 直接使用 yolo.py 的 detect_image 函式
            end_time = time.time()
            time_img = end_time - start_time  
            video_playtime.append(round(time_img, 3)) 
            result = np.asarray(image)
            cv2.putText(result, text='fps', org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            if isOutput:
                out.write(result)
        else:  
            break
            
    # 釋放資源
    vid.release()  # release input video resource
    out.release()  # release output video resource
    
    return video_playtime, video_cnt

# 偵測 Kangaroo.mp4
video_playtime, video_cnt = detect_video(yolo_model, video_path='video/Kangaroo.mp4', output_path="video/Kangaroo_bboxes.mp4")
#print('\nKangaroo.mp4 total frames:', video_cnt)  # 總共有多少 frames

avg_fps = 1/np.mean(video_playtime)
#print("Kangaroo.mp4 avg fps: %.3f" % avg_fps)  # 平均 fps


video_playtime, video_cnt = detect_video(yolo_model, video_path='video/Raccoon.mp4', output_path="video/Raccoon_bboxes.mp4")
#print('\nRaccoon.mp4 total frames:', video_cnt)  # 總共有多少 frames

avg_fps = 1/np.mean(video_playtime)
#print("Raccoon.mp4 avg fps: %.3f" % avg_fps)  # 平均 fps