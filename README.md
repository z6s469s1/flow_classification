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


training_class6.py
---
六元分類器訓練，分類出六種不同環境Streaming的flow圖片
