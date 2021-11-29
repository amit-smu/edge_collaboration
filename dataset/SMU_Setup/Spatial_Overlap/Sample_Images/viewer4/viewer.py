#!/usr/bin/python
import socket
import cv2
import numpy as np
import os
f= open('test.avi','wb')
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = ''
TCP_PORT = 5002

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()
cur= 1
datas=[]
buff1=b''
buff2=b''
frame1=0
frame2=0
g=1.5
def process(frame):
    inBlack  = np.array([0, 0, 0], dtype=np.float32)
    inWhite  = np.array([205, 255, 205], dtype=np.float32)
    inGamma  = np.array([g, g, g], dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)
    # Display the resulting frame
    img = np.clip( (frame - inBlack) / (inWhite - inBlack), 0, 255 )                            
    img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    img = np.clip( img, 0, 255).astype(np.uint8)
    img = cv2.flip(img,0)
    img = cv2.flip(img,1)
    #img = cv2.fastNlMeansDenoisingColored(img,None,3,5,7,21)  
#    img = cv2.medianBlur(img,3)
    return img 
while(1):
    if 1:
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,30)
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 2

        length = recvall(conn,16)
        stringData = recvall(conn, int(length))
        if(stringData[0]==49):
            buff1+=stringData[1:]
        if(stringData[0]==50):
            buff2+=stringData[1:]
        a = buff1.find(b'\xff\xd8')
        b = buff1.find(b'\xff\xd9')
        if(a!=-1 and b!=-1):
            frame1+=1
            #print(a,b)
            jpg = buff1[a:b+2]
            buff1=buff1[b+2:]
            ff=np.fromstring(jpg, dtype='uint8')
            #print(ff)
            img = cv2.imdecode(ff,cv2.IMREAD_COLOR)
            #print(img.shape)
            img = process(img)
            img1 = img
            res = cv2.resize(img,(600,600))
            #cv2.putText(img,'frame '+str(frame1), bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.imwrite('frame1_'+str(frame1)+'.jpg',img1)
            cv2.imshow("cam4",res)
            cv2.waitKey(3)
        a = buff2.find(b'\xff\xd8')
        b = buff2.find(b'\xff\xd9')
        if(a!=-1 and b!=-1):
            frame2+=1
            jpg = buff2[a:b+2]
            buff2=buff2[b+2:]
            img = cv2.imdecode(np.fromstring(jpg, dtype='uint8'),cv2.IMREAD_COLOR)
            img = process(img)
            img1 = img[:,:]
            #cv2.putText(img,'frame '+str(frame2), bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.imwrite('frame2_'+str(frame2)+'.jpg',img1)
            cv2.imshow("feed 2",img)
            cv2.waitKey(3)
    else:
        print("Breaking..")
        break
s.close()
