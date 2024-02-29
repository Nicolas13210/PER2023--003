#!/usr/bin/env python3

import copy
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.io import read_image
import torch.nn as nn
#from torchvision.models import MobileNet_V2_Weights
import cv2
from PIL import Image as Img
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from model import get_mobilenet_classif_1, get_classif_lstm_1, get_mobilenet_classif_2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time
# import tensorflow as tf
# import scipy.misc
#!/usr/bin/env python3
import ConvNext
import Net_RGBD
from mqtt_server import MQTTClient

MODEL_MAT=0
MODEL_NICO=1

MODEL_LOUIS=3
MODEL_NEXT = 4
MODEL_RGBD = 5

myModel = MODEL_NEXT

class VideoLSTM(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size):
        super(VideoLSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding="valid"),
            nn.BatchNorm2d(24))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding="valid"),
            nn.BatchNorm2d(36))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding="valid"),
            nn.BatchNorm2d(48))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding="valid"),
            nn.BatchNorm2d(64))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="valid"),
            nn.BatchNorm2d(128))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)

        # Adjust the output dimension to 1 since we're predicting one label for the entire sequence
        self.output = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w) 
        print("MODEL NICOO")
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = nn.functional.relu(self.layer4(x))
        x = nn.functional.relu(self.layer5(x))
        x = self.avgpool(x)
        features = self.flatten(x)
        features = features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
        output = self.output(lstm_out)

        return output.squeeze(1)  # Squeeze the output tensor to remove the extra dimension



class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3)
        self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64)
        self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same')
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)

        self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128)
        self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same')
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn9 = torch.nn.BatchNorm2d(256)
        self.bn10 = torch.nn.BatchNorm2d(256)

        self.dense1 = torch.nn.Linear(14336, 512, bias=True)
        self.dense2 = torch.nn.Linear(512, 64, bias=True)
        self.dense3 = torch.nn.Linear(64, 1, bias=True)

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
        self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
        self.relu = torch.nn.ReLU()
        
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # x.shape == (64, 113)
        x = self.deepwise_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (32, 57)
        x = self.deepwise_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pointwise_conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (16, 29)
        x = self.deepwise_conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.pointwise_conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (8, 15)
        x = self.deepwise_conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.pointwise_conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = self.maxpooling_half(x)
        x = self.dropout(x)

        # x.shape == (8, 8)
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)
        #print(x.shape)
        return x

class Model_classif_1(torch.nn.Module):
    def __init__(self):
        super(Model_classif_1, self).__init__()
        self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3)
        self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64)
        self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same')
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)

        self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128)
        self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same')
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn9 = torch.nn.BatchNorm2d(256)
        self.bn10 = torch.nn.BatchNorm2d(256)

        self.dense1 = torch.nn.Linear(8192, 512, bias=True)
        self.dense2 = torch.nn.Linear(512, 64, bias=True)
        self.dense3 = torch.nn.Linear(64, 1, bias=True)

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
        self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
        self.relu = torch.nn.ReLU()
        
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # x.shape == (512, 384)
        x = self.deepwise_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (256, 192)
        x = self.deepwise_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pointwise_conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (128, 96)
        x = self.deepwise_conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.pointwise_conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (64, 48)
        x = self.deepwise_conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.pointwise_conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (32, 24)
        x = self.deepwise_conv9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = self.pointwise_conv10(x)
        x = self.relu(x)
        x = self.bn10(x)
        x = self.maxpooling_half(x)
        x = self.dropout(x)

        # x.shape == (8, 6)
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)
        #print(x.shape)
        return x

def get_mobile_net_v2():
    model = MobileNetV2()
    if torch.cuda.is_available():
        model.cuda()
    return model

def get_video_lstm():
    model = VideoLSTM(128,128)
    if torch.cuda.is_available():
        model.cuda()
    return model

def get_convNext():
    model = ConvNext.ConvNeXt()
    if torch.cuda.is_available():
        model.cuda()
    return model

def get_RGBD():
    model = Net_RGBD.Net_RGBD()
    if torch.cuda.is_available():
        model.cuda()
    return model

def get_mobilenet_classif_1(myModel):
    # Create an instance of the model and move it to the GPU if available
    if myModel == MODEL_NICO :
        model = VideoLSTM(128,128)
    elif myModel == MODEL_LOUIS:
        model = MobileNetV2()
    elif myModel == MODEL_NEXT:
        model = ConvNext.ConvNeXt()
    else:
        model = Model_classif_1()
    if torch.cuda.is_available():
        model.cuda()  
    return model


class follow_road:
    def __init__(self,model):
        
        print('init')
        self.i = 0
        self.myModel = model
        self.lock=False
        self.paused = False
        #client.message_callback_add(client.TOPIC_STOP, self.stop_callback, qos=2)

        self.img0 = torch.zeros(1,3,64, 64)
        self.img1 = torch.zeros(1,3,64, 64)
        self.img2 = torch.zeros(1,3,64, 64)
        self.img3 = torch.zeros(1,3,64, 64)
        self.img4 = torch.zeros(1,3,64, 64)
        
        #client.message_callback_add(client.TOPIC_MODEL, self.switch_model_callback, qos=2)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        print(self.device)
        # self.model = get_mobilenet_classif_2()
        # self.model = get_mobilenet_classif_1(model)
        # self.model1 = get_video_lstm()
        # self.model1.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/LSTMaugmented.pt'))
        # print("model 1 loaded")
        self.model2 = get_mobile_net_v2()
        self.model2.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/model_louis3.pt'))
        print("model 2 loaded")
        self.model3 = get_convNext()
        self.model3.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/ConvNext1.pt'))
        print("model 3 loaded")
        self.model4 = get_RGBD()
        self.model4.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/NetRGBD.pt'))
        print(("model 4 loaded"))
        # self.model.load_state_dict(torch.load(
        #      '/root/catkin_ws/src/imredd_pkg/models/kenza/robot_3/9_labels/ckpt_44_ok.ckpt', map_location=self.device))
        # self.model.load_state_dict(copy.deepcopy(torch.load(
        #   '/root/catkin_ws/src/imredd_pkg/models/per/fullModel.pt', map_location=self.device)))
        if model == MODEL_NICO:
            self.model = self.model1
        elif model == MODEL_LOUIS:
            self.model = self.model2
        elif model == MODEL_NEXT:
            self.model = self.model3
        elif model == MODEL_RGBD:
            self.model = self.model4
        else:
            self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/model_louis.pt'))

        #print('model: /root/catkin_ws/src/imredd_pkg/models/per/fullModel.pt')
        # self.model.load_state_dict(torch.load(
        #     '/root/catkin_ws/src/imredd_pkg/models/kenza/federe/ckpt_39.ckpt', map_location=self.device))
        # load params
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # self.h0 = torch.rand((1,4,1024))

        # self.steering_gain = float(rospy.get_param("steering_gain"))
        # self.steering_dgain = float(rospy.get_param("steering_dgain"))
        # self.steering_bias = float(rospy.get_param("steering_bias"))
        self.speed = float(0.25) #0.25
        self.speed_factor = float(10) #5

        self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=30)
        self.rate = rospy.Rate(30)

        self.bridge = CvBridge()
        self.angle_last = 0.0

        self.image_rgb = rospy.Subscriber(
            "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
        #self.img_depth = None
        self.img_depth = rospy.Subscriber("/car/camera/depth/image_rect_raw", Image, self.callback, callback_args="depth")
        # self.steering_angle = rospy.Subscriber(
        #     "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
        self.dico = {'0': float(0.69), '1': float(0.55),'2': float(0.38), '3': float(0.15), '4': float(
            0.0), '5': float(-0.15),'6':float(-0.38), '7': float(-0.55), '8': float(-0.69)}
        
        
        # self.dico = {'0': float(0.69), '1': float(0.6), '2': float(0.5), '3': float(
        #     0.4), '4': float(0.3), '5': float(0.2), '6': float(0.1), '7': float(
        #         0.0), '8': float(-0.1), '9': float(-0.2), '10': float(-0.3), '11': float(
        #             -0.4), '12': float(-0.5), '13': float(-0.6), '14': float(-0.69)}
        
        # self.dico = {'0': float(0.69), '1': float(0.55), '2': float(0.3), '3': float(
        #    0.2), '4': float(-0.2), '5': float(-0.3), '6': float(-0.55), '7': float(-0.69)}
        self._run()

    def switch_model_callback(self, client, userdata, message):
        new_model = message.payload.decode('utf-8')
        while self.lock:
            print("j'attend que l'inférence se termine")
            time.sleep(0.1)
        self.lock=True
        if new_model == "1" :
            print("switch sur model nico")
            self.myModel = MODEL_NICO
            self.model = self.model1
        elif new_model == "2":
            print("switch sur model louis")
            self.myModel = MODEL_LOUIS
            self.model = self.model2
        elif new_model == "3":
            print("switch sur model  convnext")
            self.myModel = MODEL_NEXT
            self.model = self.model3
        else:
            print("Modèle non reconnu: " + new_model)
        self.lock=False

    def _run(self):
        rospy.spin()

    def get_image(self, msg):
        msg_image = msg.data

        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)


        img_1 = np.reshape(msg_image_2, (240, 424,3), order='C')

        img_1 = torch.from_numpy(img_1)
        img_1 = img_1.permute(2, 0, 1)

        width, height = transforms.functional.get_image_size(img_1)

        if self.myModel == MODEL_LOUIS or self.myModel == MODEL_NEXT:
            img_1 = transforms.functional.crop(img_1, height - 50, 0, height-50, width)
        else :
            img_1 = transforms.functional.crop(img_1, height - 60, 0, 120, width) #60
        img_1 = img_1.float()    

        if self.myModel == MODEL_NEXT:
            size = round(64)
            width_resize = round(64*424/190)
            data_transforms = transforms.Compose([
                transforms.Resize((size, width_resize), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            size=64
            data_transforms = transforms.Compose([
                transforms.Resize((size, 113), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        

        img_1 = data_transforms(img_1)

        return img_1
    
    def get_image_nico(self, msg):

        msg_image = msg.data
        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

        img_1 = np.reshape(msg_image_2, (240, 424,3), order='C')
        img_1 = torch.from_numpy(img_1)
        img_1 = img_1.permute(2, 0, 1)

        width, height = transforms.functional.get_image_size(img_1)
        img_1 = transforms.functional.crop(img_1, height - 50, 0, 120, width)
        img_1 = img_1.float()    

        size = 64
        data_transforms = transforms.Compose([
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

        img_1 = data_transforms(img_1)
        return img_1

    def preprocess_nico(self, img_rgb):  
        #return torch.tensor(self.get_image(img_rgb))
        if img_rgb is not None:
            img_rgb = self.get_image_nico(img_rgb)
            img_rgb = torch.unsqueeze(img_rgb,0)
            #print(img_rgb)
            #print(img_rgb.shape)
            return img_rgb

    def preprocess(self, img_rgb):  
        if img_rgb is not None:
            img_rgb = self.get_image(img_rgb)
            img_rgb = torch.unsqueeze(img_rgb,0)
            return img_rgb

    def get_image_rgbd(self,img_rgb,img_depth):
        img_rgb = img_rgb.data
        img_rgb = np.frombuffer(img_rgb, dtype=np.uint8)

        img_rgb = np.reshape(img_rgb, (240, 424,3), order='C')
        img_rgb = torch.from_numpy(img_rgb)
        img_rgb = img_rgb.permute(2, 0, 1)

        width, height = transforms.functional.get_image_size(img_rgb)
        img_rgb = transforms.functional.crop(img_rgb, height - 50, 0, 120, width)
        img_rgb = img_rgb.float()    

        size = 64
        width_resize = int(64 * 424 / 190)
        data_transforms = transforms.Compose([
            transforms.Resize((size, width_resize), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

        img_rgb = data_transforms(img_rgb)

        img_depth = img_depth.data
        img_depth = np.frombuffer(img_depth, dtype=np.uint16)

        img_depth = np.reshape(img_depth, (480, 848,1), order='C')
        img_depth = (img_depth / np.max(img_depth) * 255).astype(np.uint8)
        img_depth = torch.from_numpy(img_depth)
        img_depth = img_depth.permute(2, 0, 1)


        img_depth = transforms.Resize((height, width), antialias=True)(img_depth)

        width, height = transforms.functional.get_image_size(img_depth)
        img_depth = transforms.functional.crop(img_depth, height - 50, 0, 120, width)
        img_depth = img_depth.float()    

        size = 64
        width_resize = int(64 * 424 / 190)
        data_transforms = transforms.Compose([
            transforms.Resize((size, width_resize), antialias=True),
            transforms.Normalize([0.5], [0.5])
        ]) 

        img_depth = data_transforms(img_depth)

        return img_rgb,img_depth

        
    def preprocess_rgbd(self, img_rgb, img_depth):  
        if img_rgb is not None and img_depth is not None:
            img_rgb, img_depth = self.get_image_rgbd(img_rgb, img_depth)
            img_rgb = torch.unsqueeze(img_rgb,0)
            img_depth = torch.unsqueeze(img_depth,0)
            return img_rgb, img_depth

    def stop_callback(self, client, userdata, message):
        message_payload = message.payload.decode('utf-8')
        if message_payload == "true":
            self.paused = True
        elif message_payload == "false":
            self.paused = False

    def callback(self, img, args):
        if args == "rgb":
            self.img_rgb = img
        if args == "depth":
            self.img_depth = img

        #(self.img_depth is None)

        if self.img_rgb is not None:
            img_rgb = self.img_rgb
            while self.lock:
                print("j'attends que le changement de model soit effectif")
                time.sleep(0.1)
            #self.lock =True
            self.img4 = self.preprocess_nico(img_rgb)
            if self.myModel == MODEL_NICO :
                input1 = torch.cat([self.img0,self.img1,self.img2,self.img3,self.img4],dim=0)
                input1 = torch.unsqueeze(input1,dim=0)   
            elif self.myModel == MODEL_RGBD and self.img_depth is not None:
                img_rgb = self.img_rgb
                img_depth = self.img_depth
                img_rgb, img_depth = self.preprocess_rgbd(img_rgb,img_depth)
                img_rgb = img_rgb.to(self.device)
                img_depth = img_depth.to(self.device)
                output = self.model(img_rgb, img_depth)
            else:
                input1=self.preprocess(img_rgb)

            if self.myModel != MODEL_RGBD:
                input1 = input1.to(self.device)
                start_time = time.time()
                output = self.model(input1)

            self.lock=False
            # print(f'{time.time() - start_time}')

            output = output.squeeze().item()
            # output = output.item()
            # print(output)
            # # steering_label = self.model(input_1)

            # # softmax_func = torch.nn.Softmax(dim=1)
            # # softmax_steering = softmax_func(steering_label)
            # # log_steering = torch.log(softmax_steering)
            # steering_label = torch.argmax(output, dim=-1)
            # # predict = torch.max(output, dim=1)[1]
            # steering_label = torch.unsqueeze(steering_label, dim=0).item()
            # # steering = self.dico[str(steering_label)]+float(0.11)
            steering = output
            # steering = self.dico[str(steering_label)]
            speed = self.speed

            # steering = steering*self.steering_gain

        if not self.paused:
            commande = AckermannDriveStamped()
            commande.drive.speed = speed
            commande.drive.steering_angle = steering
            self.pub.publish(commande)
            self.rate.sleep()

        # Met à jour les 4 dernières images
        self.img0=self.img1
        self.img1=self.img2
        self.img2=self.img3
        self.img3=self.img4


if __name__ == "__main__":
    global client
    QOS = 0
    MQTT_HOST = "192.168.18.9"
    MQTT_PORT = 2883
    #client = MQTTClient("Robot2", MQTT_HOST, MQTT_PORT, QOS)
    #client.loop_start()
    try:
        print('start')
        rospy.init_node('FollowRoad_node', anonymous=True)
        p = follow_road(myModel)
    except rospy.ROSInterruptException:
        pass
    #client.loop_stop()