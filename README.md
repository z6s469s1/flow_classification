# flow_classification

pcap_to_img.py
---
功能為將pcap轉成灰階圖片

 欲轉成img的pcap資料夾路徑
```
inputPath=""  
```
 輸出圖片路徑
```
outputPath="/home/lab507hyc/disk4/Tor_Streaming/nonTor_img/"
```
 輸出圖片類型設定
 設定成Normal或VPN或Tor
```
flow_class="Normal"
```

training_binary.py
---
二元分類器訓練，判斷是否為Streaming的flow圖片


training_class6.py
---
六元分類器訓練，分類出六種不同環境Streaming的flow圖片
