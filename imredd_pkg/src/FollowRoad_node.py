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
# import tensorflow as tf
# import scipy.misc
#!/usr/bin/env python3

class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super(MobileNetV2, self).__init__()
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
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.output = nn.Linear(8, num_class)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = nn.functional.relu(self.layer4(x))
        x = nn.functional.relu(self.layer5(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.output(x)
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

        self.dense1 = torch.nn.Linear(23296, 512, bias=True)
        self.dense2 = torch.nn.Linear(512, 128, bias=True)
        self.dense3 = torch.nn.Linear(128, 9, bias=True)

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
        self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
        self.relu = torch.nn.ReLU()
        
        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch
    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
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


def get_mobilenet_classif_1():
    # Create an instance of the model and move it to the GPU if available
    model = MobileNetV2(1)
    if torch.cuda.is_available():
        model.cuda()

    return model


class follow_road:
    def __init__(self):
        
        print('init')
        self.i = 0
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        print(self.device)
        # self.model = get_mobilenet_classif_2()
        self.model = get_mobilenet_classif_1()
        # self.model.load_state_dict(torch.load(
        #      '/root/catkin_ws/src/imredd_pkg/models/kenza/robot_3/9_labels/ckpt_44_ok.ckpt', map_location=self.device))
        # self.model.load_state_dict(copy.deepcopy(torch.load(
        #   '/root/catkin_ws/src/imredd_pkg/models/per/fullModel.pt', map_location=self.device)))
        self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/models/per/best_model_params_BN_full.pt'))
        print('model: /root/catkin_ws/src/imredd_pkg/models/per/fullModel.pt')
        # self.model.load_state_dict(torch.load(
        #     '/root/catkin_ws/src/imredd_pkg/models/kenza/federe/ckpt_39.ckpt', map_location=self.device))
        # load params
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # self.h0 = torch.rand((1,4,1024))

        # self.steering_gain = float(rospy.get_param("steering_gain"))
        # self.steering_dgain = float(rospy.get_param("steering_dgain"))
        # self.steering_bias = float(rospy.get_param("steering_bias"))
        self.speed = float(0.25)
        self.speed_factor = float(5)

        self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=30)
        self.rate = rospy.Rate(30)

        self.bridge = CvBridge()
        self.angle_last = 0.0

        self.image_rgb = rospy.Subscriber(
            "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
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

    def _run(self):
        rospy.spin()

    def get_image(self, msg):

        # img_size = (240,424,3)
        # print('msg',msg)

        msg_image = msg.data
        # print('msg_image',msg_image)

        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

        #img_1 = np.reshape(msg_image_2, (240, 424, 3), order='C')
        #img_1 = np.reshape(msg_image_2, (424, 240, 3), order='C')

        img_1 = np.reshape(msg_image_2, (240, 424,3), order='C')

        
        Img.fromarray(img_1).save(f'/root/catkin_ws/src/imredd_pkg/models/perimage/original/img{self.i}.png')

        #img_1 = transforms.ToTensor()(img_1)
        img_1 = torch.from_numpy(img_1)
        img_1 = img_1.permute(2, 0, 1)

        self.i +=1
        
        save_image(img_1.float(), f'/root/catkin_ws/src/imredd_pkg/models/perimage/original/img{self.i}.png')
        
        #Img.fromarray(img_1).save(f'/root/catkin_ws/src/imredd_pkg/models/perimage/original/img{self.i}.png')


        width, height = transforms.functional.get_image_size(img_1)
        img_1 = transforms.functional.crop(img_1, height - 120, 0, 120, width)
        img_1 = img_1.float()    


        # print(img_1)

        # img_1 = img_1.float()



        # ret, img_1 = cv2.threshold(img_1,175,255,cv2.THRESH_BINARY)
        # img_1 = img_1[120:]

        #msg_image_2 = Img.fromarray(img_1, mode='RGB')

        #msg_image_2.save('/root/catkin_ws/src/imredd_pkg/models/perimage/original/'+ str(self.i)+".jpeg")

        """width, height = msg_image_2.size  # Get width and height
        # print(img_1.size)

        # Define the crop coordinates (left, upper, right, lower)
        crop_coords = (0, 120, width, height)  # Assuming you want to crop from height - 120 to height and from 0 to 120

        msg_image_2 = msg_image_2.crop(crop_coords)  # Crop the image"""


        #msg_image_2.save('/root/catkin_ws/src/imredd_pkg/models/perimage/before/'+ str(self.i)+".jpeg")

        # msg_image_2 = msg_image_2.float()
        

        # print(msg_image_2.shape)

        print(img_1.shape)

        size = 128
        data_transforms = transforms.Compose([
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_1 = data_transforms(img_1)
        #img_1 = img_1.float()

        #img_1 = msg_image_2

        #print(img_1.shape)

        #img_1.save('/root/catkin_ws/src/imredd_pkg/models/perimage/after/ '+ str(self.i)+".jpeg")
        self.i += 1

        

        # plt.imshow(img_1)
        # plt.show(block=False)
        # plt.show()
        # img_1[:20] = np.zeros((20,424,3))
        # copy_img = img_1 > 150
        # img_1 = copy_img*255

        return img_1

    # def preprocess(self, img_rgb,img_depth):
        # image_rgb = self.bridge.imgmsg_to_cv2(img, "bgr8")
        # image_rgb = self.get_image(img_rgb)
        # image_depth = self.get_depth_image(img_depth)
        # return image_rgb,image_depth

    def preprocess(self, img_rgb):
        
        #return torch.tensor(self.get_image(img_rgb))
        if img_rgb is not None:
            img_rgb = self.get_image(img_rgb)
            img_rgb = torch.unsqueeze(img_rgb,0)
            print(img_rgb)
            print(img_rgb.shape)
            return img_rgb
            #image_rgb = torch.tensor(self.get_image(img_rgb))
            image_rgb = transforms.ToTensor()(self.get_image(img_rgb))
            # image_rgb = torch.cat([torch.unsqueeze(torch.where(image_rgb[:,:,0] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,1] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,2] > 160, 255, 0), dim=0)], dim=0).float()

            # image_rgb[:20] = torch.zeros((20,424,3))

            #image_rgb = torch.swapaxes(image_rgb, -1, 0)
            #image_rgb = torch.swapaxes(image_rgb, -1, 1).float()

            # image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float()
            image_rgb = torch.unsqueeze(
                torch.tensor(image_rgb), dim=0).float()
            
            image_rgb = torch.mul(image_rgb,255)

            print(image_rgb)

            return image_rgb

    # def callback(self, img_rgb,img_depth):
        # speed, steering = self.model(self.preprocess(img_rgb,img_depth)).detach().float().cpu().numpy().flatten()
    def callback(self, img, args):
        if args == "rgb":
            self.img_rgb = img

        if self.img_rgb is not None:
            img_rgb = self.img_rgb

            input_1 = self.preprocess(img_rgb)
            input_1 = input_1.to(self.device)
            
            output = self.model(input_1)

            output = output.squeeze().item()
            # output = output.item()
            print(output)
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

        # print('steering categry', steering_label)
        print('steering : ', steering)
        print('speed : ', speed)

        commande = AckermannDriveStamped()
        commande.drive.speed = speed
        commande.drive.steering_angle = steering
        self.pub.publish(commande)
        self.rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('FollowRoad_node', anonymous=True)
        p = follow_road()
    except rospy.ROSInterruptException:
        pass





# #!/usr/bin/env python3

# import torchvision
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import cv2
# from PIL import Image as Img
# import numpy as np
# import rospy
# from ackermann_msgs.msg import AckermannDriveStamped
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# # from model import get_mobilenet_classif_1, get_classif_lstm_1, get_mobilenet_classif_2
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# # import scipy.misc
# #!/usr/bin/env python3

# class Model_classif_1(torch.nn.Module):
#     def __init__(self):
#         super(Model_classif_1, self).__init__()
        
#         self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3)
#         self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same')
#         self.bn1 = torch.nn.BatchNorm2d(3)
#         self.bn2 = torch.nn.BatchNorm2d(64)

#         self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64)
#         self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same')
#         self.bn3 = torch.nn.BatchNorm2d(64)
#         self.bn4 = torch.nn.BatchNorm2d(128)

#         self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128)
#         self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same')
#         self.bn5 = torch.nn.BatchNorm2d(128)
#         self.bn6 = torch.nn.BatchNorm2d(256)

#         self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
#         self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
#         self.bn7 = torch.nn.BatchNorm2d(256)
#         self.bn8 = torch.nn.BatchNorm2d(256)

#         self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
#         self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
#         self.bn9 = torch.nn.BatchNorm2d(256)
#         self.bn10 = torch.nn.BatchNorm2d(256)

#         self.dense1 = torch.nn.Linear(23296, 512, bias=True)
#         self.dense2 = torch.nn.Linear(512, 128, bias=True)
#         self.dense3 = torch.nn.Linear(128, 9, bias=True)

#         self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
#         self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
#         self.relu = torch.nn.ReLU()
        
#         self.dropout = torch.nn.Dropout(0.1)

#     '''
#     this function is made to compute prediction using the given batch
#     args:
#         x: torch tensor representing one batch of data
    
#     return:
#         x: torch tensor which contains a batch of prediction
#     '''
#     def forward(self, x):

#         # x.shape == (512, 384)
#         x = self.deepwise_conv1(x)
#         x = self.relu(x)
#         x = self.bn1(x)
#         x = self.pointwise_conv2(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (256, 192)
#         x = self.deepwise_conv3(x)
#         x = self.relu(x)
#         x = self.bn3(x)
#         x = self.pointwise_conv4(x)
#         x = self.relu(x)
#         x = self.bn4(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (128, 96)
#         x = self.deepwise_conv5(x)
#         x = self.relu(x)
#         x = self.bn5(x)
#         x = self.pointwise_conv6(x)
#         x = self.relu(x)
#         x = self.bn6(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (64, 48)
#         x = self.deepwise_conv7(x)
#         x = self.relu(x)
#         x = self.bn7(x)
#         x = self.pointwise_conv8(x)
#         x = self.relu(x)
#         x = self.bn8(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (32, 24)
#         x = self.deepwise_conv9(x)
#         x = self.relu(x)
#         x = self.bn9(x)
#         x = self.pointwise_conv10(x)
#         x = self.relu(x)
#         x = self.bn10(x)
#         x = self.maxpooling_half(x)
#         x = self.dropout(x)

#         # x.shape == (8, 6)
#         x = torch.flatten(x, start_dim=1)

#         x = self.dense1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.dense2(x)
#         x = self.relu(x)

#         x = self.dense3(x)
#         #print(x.shape)
#         return x


# def get_mobilenet_classif_1():
#     # Create an instance of the model and move it to the GPU if available
#     model = Model_classif_1()
#     if torch.cuda.is_available():
#         model.cuda()

#     return model



# class follow_road:
#     def __init__(self):
        
#         print('init')
        
#         # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = torch.device("cuda")
#         print(self.device)
#         # self.model = get_mobilenet_classif_2()
#         self.model = get_mobilenet_classif_1()
#         self.model.load_state_dict(torch.load(
#             '/root/catkin_ws/src/imredd_pkg/models/kenza/9_labels/ckpt_3_15.ckpt', map_location=self.device))
#         # load params
#         self.model = self.model.to(self.device)
#         self.model = self.model.eval()

#         # self.h0 = torch.rand((1,4,1024))

#         # self.steering_gain = float(rospy.get_param("steering_gain"))
#         # self.steering_dgain = float(rospy.get_param("steering_dgain"))
#         # self.steering_bias = float(rospy.get_param("steering_bias"))
#         self.speed = float(0.25)
#         self.speed_factor = float(5)

#         self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=30)
#         self.rate = rospy.Rate(30)

#         self.bridge = CvBridge()
#         self.angle_last = 0.0

#         self.image_rgb = rospy.Subscriber(
#             "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
#         # self.steering_angle = rospy.Subscriber(
#         #     "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
#         self.dico = {'0': float(0.69), '1': float(0.55),'2': float(0.4), '3': float(0.3), '4': float(
#             0.0), '5': float(-0.3),'6':float(-0.4), '7': float(-0.55), '8': float(-0.69)}
        
        
#         # self.dico = {'0': float(-0.65), '1': float(-0.55), '2': float(-0.45), '3': float(
#         #     -0.35), '4': float(-0.25), '5': float(-0.15), '6': float(-0.05), '7': float(
#         #         0.05), '8': float(0.15), '9': float(0.25), '10': float(0.35), '11': float(
#         #             0.45), '12': float(0.55), '13': float(0.65)}
#         # self.dico = {'0': float(0.69), '1': float(0.55), '2': float(0.3), '3': float(
#         #    0.2), '4': float(-0.2), '5': float(-0.3), '6': float(-0.55), '7': float(-0.69)}
#         self._run()

#     def _run(self):
#         rospy.spin()

#     def get_image(self, msg):

#         # img_size = (240,424,3)
#         # print('msg',msg)

#         msg_image = msg.data
#         # print('msg_image',msg_image)

#         msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

#         img_1 = np.reshape(msg_image_2, (240, 424, 3), order='C')

#         # ret, img_1 = cv2.threshold(img_1,175,255,cv2.THRESH_BINARY)
#         img_1 = img_1[120:]
#         # plt.imshow(img_1)
#         # plt.show(block=False)
#         # plt.show()
#         # img_1[:20] = np.zeros((20,424,3))
#         # copy_img = img_1 > 150
#         # img_1 = copy_img*255

#         return img_1

#     # def preprocess(self, img_rgb,img_depth):
#         # image_rgb = self.bridge.imgmsg_to_cv2(img, "bgr8")
#         # image_rgb = self.get_image(img_rgb)
#         # image_depth = self.get_depth_image(img_depth)
#         # return image_rgb,image_depth

#     def preprocess(self, img_rgb):
#         if img_rgb is not None:

#             image_rgb = torch.tensor(self.get_image(img_rgb))
#             # image_rgb = torch.cat([torch.unsqueeze(torch.where(image_rgb[:,:,0] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,1] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,2] > 160, 255, 0), dim=0)], dim=0).float()

#             # image_rgb[:20] = torch.zeros((20,424,3))

#             image_rgb = torch.swapaxes(image_rgb, -1, 0)
#             image_rgb = torch.swapaxes(image_rgb, -1, 1).float()

#             # image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float()
#             image_rgb = torch.unsqueeze(
#                 torch.tensor(image_rgb), dim=0).float()

#             return image_rgb

#     # def callback(self, img_rgb,img_depth):
#         # speed, steering = self.model(self.preprocess(img_rgb,img_depth)).detach().float().cpu().numpy().flatten()
#     def callback(self, img, args):
#         if args == "rgb":
#             self.img_rgb = img

#         if self.img_rgb is not None:
#             img_rgb = self.img_rgb

#             input_1 = self.preprocess(img_rgb)
#             input_1 = input_1.to(self.device)
            
#             output = self.model(input_1)
#             # output = output.item()
#             print(output)
#             # # steering_label = self.model(input_1)

#             # # softmax_func = torch.nn.Softmax(dim=1)
#             # # softmax_steering = softmax_func(steering_label)
#             # # log_steering = torch.log(softmax_steering)
#             steering_label = torch.argmax(output, dim=-1)
#             # # predict = torch.max(output, dim=1)[1]
#             steering_label = torch.unsqueeze(steering_label, dim=0).item()
#             # # steering = self.dico[str(steering_label)]+float(0.11)
#             steering = self.dico[str(steering_label)]
#             speed = self.speed

#             # steering = steering*self.steering_gain

#         # print('steering categry', steering_label)
#         print('steering : ', steering)
#         print('speed : ', speed)

#         commande = AckermannDriveStamped()
#         commande.drive.speed = speed
#         commande.drive.steering_angle = steering
#         self.pub.publish(commande)
#         self.rate.sleep()


# if __name__ == "__main__":
#     try:
#         rospy.init_node('FollowRoad_node', anonymous=True)
#         p = follow_road()
#     except rospy.ROSInterruptException:
#         pass




# import torchvision
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import cv2
# import PIL.Image
# import numpy as np
# import rospy
# from ackermann_msgs.msg import AckermannDriveStamped
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# #import tensorflow as tf
# #import scipy.misc
# #from subprocess import call
# #from model_classif_1 import get_mobilenet_classif_1


# class Model_classif_1(torch.nn.Module):
#     def __init__(self):
#         super(Model_classif_1, self).__init__()
        
#         self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3)
#         self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same')
#         self.bn1 = torch.nn.BatchNorm2d(3)
#         self.bn2 = torch.nn.BatchNorm2d(64)

#         self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64)
#         self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same')
#         self.bn3 = torch.nn.BatchNorm2d(64)
#         self.bn4 = torch.nn.BatchNorm2d(128)

#         self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128)
#         self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same')
#         self.bn5 = torch.nn.BatchNorm2d(128)
#         self.bn6 = torch.nn.BatchNorm2d(256)

#         self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
#         self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
#         self.bn7 = torch.nn.BatchNorm2d(256)
#         self.bn8 = torch.nn.BatchNorm2d(256)

#         self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
#         self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
#         self.bn9 = torch.nn.BatchNorm2d(256)
#         self.bn10 = torch.nn.BatchNorm2d(256)

#         self.dense1 = torch.nn.Linear(23296, 512, bias=True)
#         self.dense2 = torch.nn.Linear(512, 128, bias=True)
#         self.dense3 = torch.nn.Linear(128, 8, bias=True)

#         self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
#         self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
#         self.relu = torch.nn.ReLU()

#         self.dropout = torch.nn.Dropout(0.1)

#     '''
#     this function is made to compute prediction using the given batch
#     args:
#         x: torch tensor representing one batch of data
    
#     return:
#         x: torch tensor which contains a batch of prediction
#     '''
#     def forward(self, x):

#         # x.shape == (512, 384)
#         x = self.deepwise_conv1(x)
#         x = self.relu(x)
#         x = self.bn1(x)
#         x = self.pointwise_conv2(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (256, 192)
#         x = self.deepwise_conv3(x)
#         x = self.relu(x)
#         x = self.bn3(x)
#         x = self.pointwise_conv4(x)
#         x = self.relu(x)
#         x = self.bn4(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (128, 96)
#         x = self.deepwise_conv5(x)
#         x = self.relu(x)
#         x = self.bn5(x)
#         x = self.pointwise_conv6(x)
#         x = self.relu(x)
#         x = self.bn6(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (64, 48)
#         x = self.deepwise_conv7(x)
#         x = self.relu(x)
#         x = self.bn7(x)
#         x = self.pointwise_conv8(x)
#         x = self.relu(x)
#         x = self.bn8(x)
#         x = self.maxpooling(x)
#         x = self.dropout(x)

#         # x.shape == (32, 24)
#         x = self.deepwise_conv9(x)
#         x = self.relu(x)
#         x = self.bn9(x)
#         x = self.pointwise_conv10(x)
#         x = self.relu(x)
#         x = self.bn10(x)
#         x = self.maxpooling_half(x)
#         x = self.dropout(x)

#         # x.shape == (8, 6)
#         x = torch.flatten(x, start_dim=1)

#         x = self.dense1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.dense2(x)
#         x = self.relu(x)

#         x = self.dense3(x)
#         #print(x.shape)
#         return x


# def get_mobilenet_classif_1t():
#     # Create an instance of the model and move it to the GPU if available
#     model = Model_classif_1()
#     if torch.cuda.is_available():
#         model.cuda()

#     # Output the summary of the model
#     #print(model)
#     #summary(model, (3, 120, 424))

#     return model

# class follow_road:
#     def __init__(self):
#         print('init')
#         #self.model = torchvision.models.resnet18(pretrained=False)
#         # self.model = get_mobilenet_classif_1t()
#         self.model= get_mobilenet_classif_1t()
#         # self.model.fc =torch.nn.Linear(512,1)
#         # self.model = torch.nn.Linear(640, 1)
#         #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/test_03_02_2023_preprocessedimg.pth'))
#         #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/40mn_datasettest_23-02-2023_18-12.pth'))
#         self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/ckpt_10.ckpt'))
#         # self.device = torch.device('cuda')
#         # self.model = self.model.to(self.device)
#         # self.model = self.model.eval().half()
        
#         ##new method
#         path = '/root/catkin_ws/src/imredd_pkg/ckpt_10.ckpt'
#         torch.save(self.model.state_dict(), path) # nothing else here
#         self.model.load_state_dict(torch.load(path))
#         self.device = torch.device('cuda')
#         self.model = self.model.to(self.device)
#         self.model = self.model.eval().half()
#         ####
#         self.speed = float(1)
#         self.speed_factor = float(5)

    
#         self.rate = rospy.Rate(30)

#         self.bridge = CvBridge()
#         self.angle_last = 0.0

#         self.image_rgb = rospy.Subscriber(
#             "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
#         self.dico = {'0': float(0.7), '1': float(0.5), '2': float(0.3), '3': float(
#             0.0), '4': float(-0.3), '5': float(-0.5), '6': float(-0.7)}
#         # self.dico = {'0': float(-0.65), '1': float(-0.55), '2': float(-0.45), '3': float(
#         #     -0.35), '4': float(-0.25), '5': float(-0.15), '6': float(-0.05), '7': float(
#         #         0.05), '8': float(0.15), '9': float(0.25), '10': float(0.35), '11': float(
#         #             0.45), '12': float(0.55), '13': float(0.65)}
#         self._run()
# 	# Pre-porcessing image function
#         # self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
#         # self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        
#         # self.steering_gain=float(rospy.get_param("steering_gain"))
#         # self.steering_dgain=float(rospy.get_param("steering_dgain"))
#         # self.steering_bias=float(rospy.get_param("steering_bias"))
#         # self.speed=float(rospy.get_param("speed"))

#         # self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=10)
#         # self.rate=rospy.Rate(10)
        
#         # self.bridge = CvBridge()
#         # self.angle_last=0.0
	
#         # rospy.Subscriber("/car/camera/color/image_raw", Image, self.callback)

#         # self._run()



#     def _run(self):
#         rospy.spin()

#     # def preprocess(self, img):
#     #     image= self.bridge.imgmsg_to_cv2(img, "bgr8")
#     #     #cv2.imwrite('/root/mushr00SD/imgtest.png',image)
#     #     height , width = image.shape[:2]
#     #     image=image[int(2*height/3):height,:,:]
#     #     img_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     #     ret,img_thresh = cv2.threshold(img_grey,170,255,cv2.THRESH_BINARY)
#     #     image=cv2.Canny(img_thresh,50,150)
#     #     #cv2.imwrite('/root/mushr00SD/rettest.png',image)
#     #     image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     #     #cv2.imwrite('/root/mushr00SD/test.png',image)
#     #     image = PIL.Image.fromarray(image)
#     #     #image.save('/root/mushr00SD/testPIL.png')
#     #     '''image= self.bridge.imgmsg_to_cv2(img, "rgb8")
#     #     image = PIL.Image.fromarray(image)
#     #     width, height = image.size
#     #     image=image.crop((0,2*height/3,width,height))
#     #     print(image.size)'''
#     #     image = transforms.functional.resize(image, (120,640))
#     #     image = transforms.functional.to_tensor(image).to(self.device).half()
#     #     image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
#     #     return image[None, ...]

#     def preprocess(self, img_rgb):
#         if img_rgb is not None:

#             image_rgb = torch.tensor(self.get_image(img_rgb))
#             # image_rgb = torch.cat([torch.unsqueeze(torch.where(image_rgb[:,:,0] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,1] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,2] > 160, 255, 0), dim=0)], dim=0).float()

#             # image_rgb[:20] = torch.zeros((20,424,3))

#             image_rgb = torch.swapaxes(image_rgb, -1, 0)
#             image_rgb = torch.swapaxes(image_rgb, -1, 1).float()

#             # image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float()
#             image_rgb = torch.unsqueeze(
#                 torch.tensor(image_rgb), dim=0).float()

#             return image_rgb

#     def get_image(self, msg):

#         # img_size = (240,424,3)
#         # print('msg',msg)

#         msg_image = msg.data
#         # print('msg_image',msg_image)

#         msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

#         img_1 = np.reshape(msg_image_2, (240, 424, 3), order='C')

#         img_1 = img_1[120:]

#         return img_1
    
#     def callback(self, img, args):
#         if args == "rgb":
#              self.img_rgb = img

#         if self.img_rgb is not None:
#             img_rgb = self.img_rgb

#             input_1 = self.preprocess(img_rgb)

#             steering_label = self.model(input_1)
#             # steering_label = self.model(input_1)

#             steering_label = torch.argmax(steering_label, dim=-1)
#             steering_label = torch.unsqueeze(steering_label, dim=0).item()
#             # steering = self.dico[str(steering_label)]+float(0.11)
#             steering = self.dico[str(steering_label)]
#             speed = self.speed

#             # steering = steering*self.steering_gain

#         print('steering categry', steering_label)
#         print('steering : ', steering)
#         print('speed : ', speed)

#         # commande = AckermannDriveStamped()
#         # commande.drive.speed = speed * self.speed_factor
#         # commande.drive.steering_angle = steering
#         # self.pub.publish(commande)
#         self.rate.sleep()

#     # def callback(self, img):
#     #     steering = self.model(self.preprocess(img)).detach().float().cpu().numpy().flatten()
#     #     print('steering : ',steering)
#     #     print('len :' ,len(steering))
#     #     print('steering_gain:' ,self.steering_gain)
#     #     commande=AckermannDriveStamped()
#     #     commande.drive.speed=self.speed
#     #     commande.drive.steering_angle=steering*self.steering_gain
#     #     print("commande: ",commande.drive.steering_angle)
#     #     self.pub.publish(commande)
#     #     self.rate.sleep()

# if __name__=="__main__":
#     try:
#         rospy.init_node('FollowRoad_node', anonymous=True)
#         p = follow_road()
#     except rospy.ROSInterruptException:
#         pass

