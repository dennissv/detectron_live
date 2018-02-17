#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:37:31 2018

@author: dennis
"""

import cv2
import time
import numpy as np
import socket
import sys
import pickle
from threading import Thread
import struct

class VideoStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):
		Thread(target=self.update, args=()).start()
		return self
    
	def update(self):
		while True:
			if self.stopped:
				return
			self.grabbed, self.frame = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True

#if __name__ == "__main__":
stream = VideoStream()
stream.start()
cv2.namedWindow('Live')
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('ec2-52-17-108-146.eu-west-1.compute.amazonaws.com',9111))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
data_r = ""
payload_size = struct.calcsize("L")
frame_nr = 1
while True:
    checkpoint = time.time()
    frame = stream.read()
    result, cframe = cv2.imencode('.jpg', frame, encode_param)
    k = cv2.waitKey(1)
#    if k%256==27: break
    data = pickle.dumps(cframe)
    clientsocket.sendall(struct.pack("L", len(data))+data)
    print('Sent packet', time.time()-checkpoint)
    
    checkpoint = time.time()
    while len(data_r) < payload_size:
        data_r += clientsocket.recv(40960)
    packed_msg_size = data_r[:payload_size]
    data_r = data_r[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    while len(data_r) < msg_size:
        data_r += clientsocket.recv(40960)
    frame_data = data_r[:msg_size]
    data_r = data_r[msg_size:]
    print('Recieved packet', time.time()-checkpoint)
    
    checkpoint = time.time()
    im=pickle.loads(frame_data)[1]
    im = cv2.resize(cv2.imdecode(im,1), (640, 480))
    combined = np.concatenate((frame, im), axis=1)
    cv2.imshow('Live', combined)
    print('Finished frame {}'.format(frame_nr), time.time()-checkpoint)
    print('-'*80)
    frame_nr += 1
cv2.destroyAllWindows()