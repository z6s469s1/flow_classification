# flow_classification

pcap_to_img.py
---
功能為將pcap轉成灰階圖片

欲轉成img的pcap資料夾路徑
```
inputPath="/home/lab507hyc/disk4/Tor_Streaming/Pcaps/nonTor/"
```
輸出圖片路徑
```
outputPath="/home/lab507hyc/disk4/Tor_Streaming/nonTor_img/"
```
輸出圖片類別設定
設定成Normal或VPN或Tor
會將輸出的圖片檔案名稱增加此類別字串
方便在訓練模型當作ground truth使用
```
flow_class="Normal"
```
可以在pcap檔名偵測web_list的網站類型子串
若偵測到會將此網站類型輸出到圖片檔案名稱
```
web_list=["youtube","seventeen","twitch","aim_chat","bittorrent","email","facebook_audio","facebook_chat","ftps","hangouts_audio","hangouts_chat","icq_chat","netflix","sftp","skype_audio","skype_chat","skype_files","spotify","vimeo","voipbuster"]
```
取多少packet製作一張圖片
```
PACKET_RANGE=200
```
輸出圖片大小
```
IMG_WIDTH=200
```

training_binary.py
---
二元分類器訓練，判斷是否為Streaming的flow圖片

訓練函數
```
training(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TRAINING_DATA_PATH,TESTING_DATA_PATH)
```
評估效能函數
```
evaluation(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TESTING_DATA_PATH,EVALUATION_MODEL_PATH)
```

各參數設定:
分類類別設定
```
CLASS_DIC={0:"nonStreaming",1:"isStreaming"}
```
輸入圖片格式設定
```
IMG_SHAPE=(200,200,1)
```
判斷類別數量
```
NUM_CLASS=1
```
訓練集路徑
```
TRAINING_DATA_PATH="dataset/training/"
```
測試集路徑
```
TESTING_DATA_PATH="dataset/testing/"
```
欲評估模型路徑
```
EVALUATION_MODEL_PATH="record/weights-improvement-30-0.98.hdf5"
```


training_class6.py
---
六元分類器訓練，分類出六種不同環境Streaming的flow圖片

訓練函數
```
training(class_dic,img_shape,num_class,training_data_path_VPN,training_data_path_Tor,training_data_path_Normal,testing_data_path_VPN,testing_data_path_Tor,testing_data_path_Normal)

```
評估效能函數
```
evaluation(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TESTING_DATA_PATH_VPN,TESTING_DATA_PATH_TOR,TESTING_DATA_PATH_NORMAL,EVALUATION_CNN_PATH,EVALUATION_VGG16_PATH)
```

各參數設定:
分類類別設定
```
CLASS_DIC={0:"Normal_nonStreaming",1:"Normal_isStreaming",2:"VPN_nonStreaming",3:"VPN_isStreaming",4:"Tor_nonStreaming",5:"Tor_isStreaming"}
```
輸入圖片格式設定
```
IMG_SHAPE=(200,200,1)
```
判斷類別數量
```
NUM_CLASS=6
```
訓練集路徑
```
TRAINING_DATA_PATH_VPN="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_VPN/dataset/training/"
TRAINING_DATA_PATH_TOR="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Tor/dataset/training/"
TRAINING_DATA_PATH_NORMAL="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Normal/dataset/training/"
```
測試集路徑
```
TESTING_DATA_PATH_VPN="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_VPN/dataset/testing/"
TESTING_DATA_PATH_TOR="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Tor/dataset/testing/"
TESTING_DATA_PATH_NORMAL="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Normal/dataset/testing/"
```
欲評估模型路徑
```
EVALUATION_CNN_PATH="record/CNN-weights-improvement-07-0.80.hdf5"
EVALUATION_VGG16_PATH="record/VGG16-weights-improvement-10-0.86.hdf5"

```
