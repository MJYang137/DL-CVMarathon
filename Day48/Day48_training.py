#!/usr/bin/env python
# coding: utf-8

# # 將自己的 google drive 空間 mount 到 colab 環境
# 
# - 若在 colab 環境，請執行下面程式，點開連結及選取自己的 gmail，將許可碼拷貝貼在格子裡

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# # 安裝相關套件

# In[ ]:


get_ipython().system('pip install contextlib2')
get_ipython().system('pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"')
get_ipython().system('pip install toolz --upgrade')


# # 設定環境
# 
# - 由於 colab 每個 notebook 環境皆獨立，必須重新安裝套件

# In[ ]:


get_ipython().run_line_magic('cd', '/content/gdrive/My\\ Drive/models/research')
get_ipython().system('mkdir train eval')
#%set_env PYTHONPATH=`pwd`:`pwd`/slim


# In[ ]:


get_ipython().system('python setup.py install')


# In[ ]:


get_ipython().run_line_magic('cd', 'slim')
get_ipython().system('python setup.py install')
get_ipython().run_line_magic('cd', '..')


# # 下載資料 (pets) ; 將資料轉換成 tfrecord
# 
# - 除了 pets, 也可以下載 coco / open images / pascal VOC 資料集並用已經提供的 tfrecord 轉換工具作轉換
#     - 轉換程式在 `object_detection/dataset_tools/` 裡
# - 也可以用定義自己的資料集及資料格式，再客製轉換程式將資料轉換成 tfrecord

# In[ ]:


# 下載資料
get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
# 解壓縮
get_ipython().system('tar -xvf images.tar.gz')
get_ipython().system('tar -xvf annotations.tar.gz')


# In[ ]:


get_ipython().system('python object_detection/dataset_tools/create_pet_tf_record.py  --label_map_path=object_detection/data/pet_label_map.pbtxt  --data_dir=/content/gdrive/My\\ Drive/models/research  --output_dir=/content/gdrive/My\\ Drive/models/research')
get_ipython().system('ls *.record*')


# # 到 [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 下載 ssd_mobilenet_v1_coco 預訓練模型權重
# 
# - 此範例以 ssd_mobilenet_v1 作參考，可以試用別的 (如 ssd_mobilenet_v2)

# In[ ]:


# 下載模型權重
get_ipython().system('wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz')
# 解壓縮
get_ipython().system('tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz')
get_ipython().system('cp ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.* .')


# # 調整訓練設定及超參數
# 
# - 設定檔皆在 object_detection/samples/configs/ 裡，選擇對應的模型設定檔
# - 此範例為了簡化，故用指令對設定檔作調整，實際上可以直接打開檔案作調整
# - 一些必備的調整如
#     - `PATH_TO_BE_CONFIGURED` 為調整檔案路徑位置
#     - `mscoco_label_map.pbtxt` & `mscoco_train.record` & `mscoco_val.record` 為調整資料集
# - 其他超參數也可以作調整，是需求而定

# In[ ]:


get_ipython().system('cp object_detection/samples/configs/ssd_mobilenet_v1_coco.config .')
# 修改檔案位置
get_ipython().system('sed -i "s|PATH_TO_BE_CONFIGURED|\\.|g" ssd_mobilenet_v1_coco.config')
# 修改資料集
get_ipython().system('sed -i "s|mscoco_label_map.pbtxt|object_detection/data/pet_label_map.pbtxt|g" ssd_mobilenet_v1_coco.config')
get_ipython().system('sed -i "s|mscoco_train.record|pet_faces_train.record|g" ssd_mobilenet_v1_coco.config')
get_ipython().system('sed -i "s|mscoco_val.record|pet_faces_val.record|g" ssd_mobilenet_v1_coco.config')
# 由於 pets 最終只會產生 10 份 tfrecord, 因此這裏稍作修改
get_ipython().system('sed -i "s|00100|00010|g" ssd_mobilenet_v1_coco.config')


# # 執行訓練

# In[ ]:


get_ipython().system('python object_detection/model_main.py  --logtostderr  --pipeline_config_path=ssd_mobilenet_v1_coco.config  --train_dir=train')


