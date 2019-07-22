import shutil
from collections import Counter
import sys
sys.path.append("./src")
import cv2
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.system('rm ~/yhr/in_out_moments/*')
os.system('rm ~/yhr/filtered_out/*')
os.system('rm ~/yhr/id_deliver/tmp2/*.jpg')
import threading
import pdb
import pickle
import datetime
import HKIPcamera
import numpy as np
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import socket
import os.path as osp
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature
from numpy import average, linalg, dot
from torchvision import transforms as T
import torch.nn.functional as F

import touching_ai_develop
print(touching_ai_develop.__file__)
from touching_ai_develop.msg import PayState
import rospy
from std_msgs.msg import Header
import threading


import argparse
parser = argparse.ArgumentParser(description="ID Deliver")
parser.add_argument('-a', '--action', type=str, default='add', choices=['add', 'write'])
args = parser.parse_args()


fall_time=time.time()

class MsgSubscribeClass(threading.Thread):
    '''
    Paystate [payState0, payState1] meanings: [0, 0] in an shelf, [1, 0] paying, [0, 1] finished paying, [1, 1] meaningless.
    '''
    
    def __init__(self):
        threading.Thread.__init__(self)
        self._rate = rospy.Rate(30)
        self._paySateMsg = PayState()
        rospy.Subscriber("/PayState", PayState, self.callback_pay_state)

    def callback_pay_state(self, msg):
        self._paySateMsg = msg

    def getPayState(self):
        return self._paySateMsg



    
class HKCamera(object):
    def __init__(self, ip, name, pw):
        self._ip = ip
        self._name = name
        self._pw = pw
        HKIPcamera.init(self._ip, self._name, self._pw)

    def getFrame(self):
        # HKIPcamera.init(self._ip, self._name, self._pw)
        frame = HKIPcamera.getframe()
        return frame



def playsound(SOUND_IP):
    cmd='sshpass -p "hri" ssh hri@{} "python play.py"'.format(SOUND_IP)
    os.system(cmd)

def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    x = fea1
    y = fea2
    #m, n = x.size(0), y.size(0)
    m,n=1,1
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return torch.sum(dist)
	
def jieduan(img,left,top,right,bottom):
    #left = 1920-left
    #right = 1920-right
    imgg=np.zeros((bottom-top,right-left,3))
    #for i in range(top,bottom):
    #  for j in range(left,right):
    #    for k in range(3):
    #      imgg[i-top][j-left][k] = img[i][j][k]
    #imgg = img[top:bottom+1, left:right+1, :]
    imgg = img[top:bottom, left:right, :]
    #print(imgg.shape)
    cv2.imwrite('temp.jpg',imgg)
    #pdb.set_trace()
    
    return imgg


def compute_distance(frame, b_b):
    left=int((b_b[0] - b_b[2]/2.0) * size[0])
    top=int((b_b[1]- b_b[3]/2.0) * size[1])
    right=int((b_b[0] + b_b[2]/2.0) * size[0])
    bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
    img1 = jieduan(frame,left,top,right,bottom)
    img = np.transpose(img1, (2,0,1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1
    #for feature2,filename in shujuku:

    for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
    return minsim
	

def new_people_shujuku(feature, new_name):
    if checkout_flag:
        print('checkout_flag == True')
        return 
    if shujuku.has_key(new_name):
      shujuku[new_name].append(feature)
    else:
      shujuku[new_name] = []
      shujuku[new_name].append(feature)
      print("in:"+str(new_name))

def del_people_shujuku(new_name):
    if shujuku.has_key(new_name):
      del shujuku[new_name]
      print('No. {} check out!'.format(new_name) )
    for img_file in os.listdir('./tmp2'):
        id_name = img_file.split('_')[-1][:-4]
        print(id_name)
        if id_name == str(new_name):
            os.system('rm ./tmp2/' + img_file)
   
####
height = 256
width = 128
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([
    #T.RectScale(height, width),
    T.ToTensor(),
    normalizer,
])
def preprocess(img):
    img = cv2.resize(img, (128,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #img[:,:,::-1]
    img = test_transformer(img)
    img = torch.unsqueeze(img, 0)
    #print(img.shape)
    return img
####
global print_frame
print_frame = 0
def update_shujuku(frame, b_b, new_name, newout):
    global print_frame
    global x_history
    left=int((b_b[0] - b_b[2]/2.0) * size[0])
    top=int((b_b[1]- b_b[3]/2.0) * size[1])
    right=int((b_b[0] + b_b[2]/2.0) * size[0])
    bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
    center_x = b_b[0]*size[0]
    #print(right- left)
    print('bottom:{}'.format(bottom))
    print('width:{}'.format(right-left))
    print('height:{}'.format(bottom-top))
    if top<0 or top>=bottom or left>=right or left<0 or right<0 or bottom<0: #or left<300: 
      return False
    if right - left < 110:
      #print('bounding box width too small')
      return False
    if bottom - top < 150:
      #print('bounding box height too small')
      return False
  
    #if bottom < 480:
    #  return False
    print('center_x:{}'.format(center_x))
    x_history.append(center_x)
    print('x_history:')
    print(x_history)
    img1 = jieduan(frame,left,top,right,bottom)

    print_frame += 1
    img = preprocess(img1)

    #print(left)
    #print(top)
    #print(right)
    #print(bottom)
    #print(img.shape)
    feature = extract_cnn_feature(model, img.cuda())
    if len(shujuku):
      minsim = -1
      id_name = 'new'
      #rentidir = '/home/tujh/renti/'
      #for feature2,filename in shujuku:
      for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                id_name = query


      print('minsim:{}'.format(minsim))
      print('new_in:' + str(new_in))

      new_people_shujuku(feature, new_name)
      cv2.imwrite('./tmp2/' + str(print_frame) +'_' + str(new_name) +'.jpg',
                  img1)
      features_at_door.append(feature)

      return False
    else:
      new_people_shujuku(feature, new_name)
      features_at_door.append(feature)
      cv2.imwrite('./tmp2/' + str(print_frame) +'_' + str(new_name) +'.jpg', img1)  
      return False
    #cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    #cv2.imshow('Fall detection',frame)

def shujuku_print(dic):
    print('shujuku:')
    for i in dic:
      print(i)


def get_out(last_id):
    global features_at_door

    del_people_shujuku(last_id)
    print('last_id:{}'.format(last_id))
    if len(shujuku) == 1:
        del_people_shujuku(list(shujuku.keys())[0])
        return

    potential_ids = []
    for feature in features_at_door:
      minsim = -1
      id_name = 'new'
      for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                id_name = query
      potential_ids.append(id_name)

    counter = Counter(potential_ids)
    #print(counter.most_common())
    id_to_del = counter.most_common()[0][0]
    #print('potential ids:')
    #print(potential_ids)
    #print('id_to_del:{}'.format(id_to_del))
    del_people_shujuku(id_to_del)



#open the input video file
camera = HKCamera('192.168.0.8', 'admin', 'a1234567')
#cap=cv2.VideoCapture('rtsp://admin:a1234567@192.168.0.7:554/11')
#cap_door = cv2.VideoCapture('rtsp://admin:a1234567@192.168.0.13:554/11')
#logs_dir = '/home/yehr/PCB_RPP_1028/market-1501-Exper33/RPP/'
logs_dir = os.path.abspath('./src')
num_features = 256
num_classes = 751
T = 1
dim = 256
dropout = 0.5

model = models.create('resnet50_rpp', num_features=num_features, dropout=dropout, num_classes=num_classes, cut_at_pooling=False, FCN=True, T=T, dim=dim)
model = model.cuda()
checkpoint = load_checkpoint(osp.join(logs_dir, 'cvs_checkpoint_0107.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

#get fps the size 
#fps = input_movie.get(cv2.CAP_PROP_FPS)  
#size = (int(cap_door.get(cv2.CAP_PROP_FRAME_WIDTH)),   
#        int(cap_door.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
#print(size)
#define the type of the output movie  
#output_movie = cv2.VideoWriter('out_cs4.avi', cv2.VideoWriter_fourcc(*'XVID'), fps)

# load network and weights
m = Darknet('src/cfg/yolov3.cfg')
m.load_weights('src/yolov3.weights')

use_cuda = 1
if use_cuda:
    m.cuda()
    
num_classes = 80
if num_classes == 20:
    namesfile = 'src/data/voc.names'
elif num_classes == 80:
    namesfile = 'src/data/coco.names'
else:
    namesfile = 'src/data/names'
class_names = load_class_names(namesfile)

res=[]
frame_number=0

#--datasets
if args.action == 'add':
  try:
    with open('/data/reid/renti/data.pkl','r') as pkl:
      shujuku = pickle.load(pkl)
      person_id = int(max(shujuku.keys())) +1
  except:
    shujuku = {}      
    person_id = 0
    
elif args.action == 'write':
    shujuku = {}
    person_id = 0 


new_end = 3  #detection time end
new_start = 0  #compute detection time,new_start += 1 if has detected person
new_in = False  #if detect the new people or not
new_out = False #according to the similarity to decide if delete
detect_person = False ##if has detect a person
now_time_str = 'nothing'
shujuku_length= len(shujuku)
if_out = False
global x_history
global features_at_door
x_history = []
features_at_door = []

rospy.init_node('MsgSubNode', anonymous=True)
threadSubMsg = MsgSubscribeClass()
threadSubMsg.setDaemon(True)
threadSubMsg.start()
checkout_flag = False

if_write = True
if_online = True
address = ('192.168.0.176', 1345)
def build_connection():
    print('connection success!')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)
    return s

if if_online:
    s = build_connection()

left_th = 140
right_th = 160
    
print('READY:)')
while True:
    # Grab a single frame of video
    #ret, frame = cap.read()
    ret_door = True
    frame_door = np.array(camera.getFrame())
    frame_door = frame_door[0:600,750:1050]
    size = (frame_door.shape[1], frame_door.shape[0])
    #cv2.imwrite('before_cropped_frame.jpg', frame_door)

    #print(frame_door.shape)
    #print(size)
    #cv2.imshow('door',frame_door)
    #print('new_start:{}'.format(new_start))
    #print('here')
    cv2.imwrite('cropped_frame.jpg', frame_door)
    if ret_door == True:
        if frame_number%12 == 0:
            #with open(os.path.join(logs_dir, 'data.pkl'),'w') as pkl:
            if if_write == True:
                with open('/data/reid/renti/data_pre.pkl','w') as pkl:
                    print("ijn")
                    pickle.dump(shujuku, pkl)
                shutil.copy('/data/reid/renti/data_pre.pkl', '/data/reid/renti/data.pkl')
        if frame_number%12 == 6:
            if if_write == True:
                with open('/data/reid/renti/data_pre_bu.pkl','w') as pkl: # backup
                    pickle.dump(shujuku, pkl)
                shutil.copy('/data/reid/renti/data_pre_bu.pkl', '/data/reid/renti/data_bu.pkl')
            
            #print('saved here!')
            else:
                print('NOT WRITING DATABASE!')
            print(shujuku.keys())
            
        frame_number += 1
        #print(frame_number)
        #frame_door = frame_door[:,300:]
        cv2.imwrite('frame.jpg',frame_door)
        #cv2.rectangle(frame_door, (700, 0), (1000,720), (255, 0, 0), 2)        
        #cv2.imshow('door',frame_door)
        #cv2.waitKey(1)
        #new_start+=1
        wh_ratio = frame_door.shape[1] / frame_door.shape[0]
        # Quit when the input video file ends
        if type(frame_door)!=np.ndarray:  # is frame readed is null,   go to the next loop
            print('********************this frame is null')
            continue
        # detect per 8 frame
        #if frame_number%8==1 or frame_number%8==2 or frame_number%8==3 or frame_number%8==4 or frame_number%8==5 or frame_number%8==6 or frame_number%8==7:
            #cv2.imshow('Fall detection',frame)
        #    continue

        #sized = cv2.resize(frame, (m.width, m.height))
        #sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized_door = cv2.resize(frame_door, (m.width, m.height))
        sized_door = cv2.cvtColor(sized_door , cv2.COLOR_BGR2RGB)
        if frame_number % 1 ==0 and detect_person==False:
            new_start = 0
            # check pay state
            paystatemsg = threadSubMsg.getPayState()
            checkout_person_id = paystatemsg.personID
            pay_state1 = paystatemsg.payState1
            pay_state2 = paystatemsg.payState2
            #print('personID = {}, payState1 = {}, payState2 = {}'.format(person_id, pay_state1, pay_state2))

            # dont use checkout signal temporarily
            #if [pay_state1, pay_state2] == [0, 1]:
            #    del_people_shujuku(checkout_person_id)
            #    checkout_flag = True
            #
                   
            r_door = []
            r_door = do_detect(m,sized_door,0.5,0.4,use_cuda)
            res_door = []
            for item in r_door:
                if class_names[item[6]]=='person' or class_names[item[6]]=='dog' or class_names[item[6]]=='cat' or class_names[item[6]]=='horse':
                    res_door.append(item)
            if len(res_door)==1:  # one person per loop
                detect_person = True
                new_start = 0
        #if frame_number %120 == 0 and detect_person == False:
        #    print("#---------no person detected---------")

        #start=time.time()
        #r = do_detect(m, sized, 0.5, 0.4, use_cuda)
        r_door = []

        if new_start%1==0 and detect_person == True: #if person detected, we deliver the person ID and collect features every  1 frame
          #print("#---------one person---------")
          r_door = do_detect(m,sized_door,0.5,0.4,use_cuda)
          if len(x_history) == 1:
              if if_online:
                  s = build_connection()
                  if x_history[0] < left_th:
                      s.send('Bye')
                  else:
                      s.send('Welcome')
          
          if new_start >= new_end:
            new_start=0
            if len(x_history) <= 1:
                print('Less than one frame captured! x_history:{}'.format(x_history))

            elif x_history[0] < left_th and x_history[-1] > right_th:
                #if if_online:
                #    s = build_connection()
                #    s.send('Bye')
                print('Customer is getting out!')
                get_out(person_id)
                person_id = person_id - 1
            elif x_history[0] < left_th and x_history[-1] < left_th: # this guy wanted to go out but withdraw.
                #if if_online:
                #    s = build_connection()
                #    s.send('Bye')
                
                del_people_shujuku(person_id)
                #person_id = person_id - 1
                print('Customer gives up leaving!')

            elif x_history[0] > right_th and x_history[-1] < left_th:
                #if if_online:
                #    s = build_connection()                    
                #    s.send('Welcome')
                print('Customer is getting in!')
                    
            elif x_history[0] > right_th and  x_history[-1] > right_th: # this guy wanted to get in but withdraw.
                #if if_online:
                #    s = build_connection()
                #    s.send('Welcome')
                
                del_people_shujuku(person_id)
                #person_id = person_id-1
                print('Customer gives up entering!')

            else:
                print('Cannot tell in or out! x_history:{}'.format(x_history))
                
            x_history = []
            features_at_door = []
            
            
            if new_in == True and len(shujuku)>shujuku_length:
              #checkout_flag = False  
              person_id += 1
              shujuku_length = len(shujuku)
              if_out = False
            elif new_in == True and len(shujuku)<shujuku_length and if_out == False:
              person_id+=1
              shujuku_length = len(shujuku)
              if_out = True
            new_in = False
            new_out = False
            detect_person = False
            shujuku_print(shujuku)
            #print(time.time())
            #print(time.time())
        else:
          #cv2.imshow('frm',frame_door)
          new_start+=1
          continue
		
        res_door = []

        for item in r_door:
            if class_names[item[6]]=='person' or class_names[item[6]]=='dog' or class_names[item[6]]=='cat' or class_names[item[6]]=='horse':
                res_door.append(item)

        if len(res_door) == 1:
          #print('res_door:')
          #print(res_door)
          #print('runhere')
          new_start=0  # wait at the last frame that detected human
          for people in res_door:
              new_people = people
              #print(new_out)
              if new_out == True:
                  continue
              elif (len(new_people)>0) and new_in == False:
                  #now_time = datetime.datetime.now()
                  #now_time_str = datetime.datetime.strftime(now_time,'%H:%M:%S')
                  new_out=update_shujuku(frame_door, new_people, person_id , new_out)
                  new_in = True
              elif (len(new_people)>0) and new_out == False:
                  new_out=update_shujuku(frame_door,new_people,person_id ,new_out)
        new_start+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

s.close()        
# All done!
#cap_door.release()
cv2.destroyAllWindows()
