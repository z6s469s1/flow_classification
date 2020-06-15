import os

import scapy
from scapy.all import *
from scapy.utils import PcapReader

import cv2
import numpy as np

inputPath="/home/lab507hyc/disk4/Tor_Streaming/Pcaps/nonTor/"
outputPath="/home/lab507hyc/disk4/Tor_Streaming/nonTor_img/"
flow_class="Normal"
web_list=["youtube","seventeen","twitch","aim_chat","bittorrent","email","facebook_audio","facebook_chat","ftps","hangouts_audio","hangouts_chat","icq_chat","netflix","sftp","skype_audio","skype_chat","skype_files","spotify","vimeo","voipbuster"]
PACKET_RANGE=200
IMG_WIDTH=200
RAWDATA_SIZE=IMG_WIDTH*IMG_WIDTH




def main():
    
    if os.path.isdir(outputPath) == False:
        os.mkdir(outputPath)
    
    #read all pcaps from the path of input 
    pcap_names = os.listdir(inputPath)
    
    num_img=0
    for num_pcap,pcap_name in enumerate(pcap_names):
        
        
        if ("pcap" in pcap_name) and (not "pcapng" in pcap_name):
            print("-"*80)
        
            print("loading:"+pcap_name)
            packets = rdpcap(inputPath+pcap_name)
            print(pcap_name+" have loaded.")
        
            row_data=[]
            for num_packet,packet in enumerate(packets):
            
                #filter some not necessary packets and write payloads into row_data[]
                if (TCP in packet or UDP in packet) and (Raw in packet):  
                    payload = str(packet[Raw])
                    for i in range(len(payload)):
                        row_data.append(ord(payload[i]))
            
                #output img from row_data[] and if row_data[] not enough then fill zero        
                if (num_packet%PACKET_RANGE==0) and (num_packet!=0):     
                    img=[]
                    if len(row_data)>=RAWDATA_SIZE:
                        img=row_data[0:RAWDATA_SIZE]
                    else:
                        img=row_data
                        dif=RAWDATA_SIZE-len(img)
                        for i in range(dif):
                            img.append(0)
                        
                    web_class=""
                    for web_name in web_list:
                        if web_name in pcap_name:
                            web_class=web_name
                            break    
                    if web_class=="":
                        web_class=pcap_name[:-5]
                        
                
                    img=np.array(img)
                    img=img.reshape((IMG_WIDTH,IMG_WIDTH))
                    img_filename=flow_class+"_"+web_class+"_"+str(num_img)+".png"
                    cv2.imwrite(outputPath+img_filename, img)
                    num_img=num_img+1
                    row_data=[]
            
                #print info
                pcap_info="pcap:"+str(num_pcap+1)+"/"+str(len(pcap_names))
                packet_info="packet:"+str(num_packet+1)+"/"+str(len(packets))
                print(pcap_info+"  "+packet_info+"  img:"+str(num_img))   
                
            print("-"*80)
                         
                            



if __name__=="__main__":
    main()
