#!/usr/bin/env python3

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from PIL import Image as Img
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model_copy import get_mobilenet_2
# from model_classify import get_mobilenet_classif_2
from imredd_pkg.src.model_old import get_mobilenet_classif_2, get_mobilenet_classif_1
# import tensorflow as tf
# import scipy.misc
#!/usr/bin/env python3


class follow_road:
    def __init__(self):
        print('init with classif_1')
        # self.model = get_mobilenet_classif_1().cuda()
        self.model=get_mobilenet_classif_1()
        # self.model.fc = torch.nn.Linear(512, 1)
        #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/test_03_02_2023_preprocessedimg.pth'))
        self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/best_model_classif.ckpt'))
        # self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/ckpt_189.ckpt'))
        # state_dict = torch.load('/root/catkin_ws/src/imredd_pkg/ckpt_189.ckpt')

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()

        # for key, value in state_dict.items():
        #     key = key[1:] # remove `att.`
        #     new_state_dict[key] = value

        # # load params
        # self.model.load_state_dict(new_state_dict)
        
        # self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/ckpt_189.ckpt'))
        self.device = torch.device('cuda:0')
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # self.steering_gain=float(rospy.get_param("steering_gain"))
        # self.steering_dgain=float(rospy.get_param("steering_dgain"))
        # self.steering_bias=float(rospy.get_param("steering_bias"))
        # self.speed=float(rospy.get_param("speed"))
        # self.speed_factor = float(2)

        self.steering_class = [0.65, 0.55, 0.45, 0.35, 0.25, 0.15,
                               0.05, -0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65]

        self.pub = rospy.Publisher(
            "/car/mux/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)
        self.rate = rospy.Rate(10)

        self.bridge = CvBridge()
        self.angle_last = 0.0

        self.image_rgb = rospy.Subscriber(
            "/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
        # self.image_g = rospy.Subscriber("/car/camera/depth/image_rect_raw", Image, self.callback, callback_args="depth")

        self._run()

    def _run(self):
        rospy.spin()

    def get_image(self, msg):

        # img_size = (240,424,3)
        # print('msg',msg)

        msg_image = msg.data
        # print('msg_image',msg_image)

        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)
        msg_image_2 = msg_image_2[120:]
        img_1 = np.reshape(msg_image_2, (120, 424, 3), order='C')

        return img_1

    def get_depth_image(self, msg):

        # img_size = (480,848,2)

        msg_image = msg.data
        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

        img_ = np.reshape(msg_image_2, (480, 848, 2), order='C')
        img = img_[:, :, 0]

        return img

    # def preprocess(self, img_rgb,img_depth):
        # image_rgb = self.bridge.imgmsg_to_cv2(img, "bgr8")
        # image_rgb = self.get_image(img_rgb)
        # image_depth = self.get_depth_image(img_depth)
        # return image_rgb,image_depth

    def preprocess(self, img_rgb, img_depth=None):
        # if img_rgb is not None and img_depth is not None:
        if img_rgb is not None:
            image_rgb = self.get_image(img_rgb)
            # image_depth = self.get_depth_image(img_depth)
            # image_rgb, image_depth = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float(), torch.unsqueeze(torch.tensor(image_depth), dim=0).float()
            image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float()

            # image_depth = torch.unsqueeze(torch.tensor(image_depth), dim=0).float().cuda()

            image_rgb = torch.swapaxes(image_rgb, -1, 1)
            image_rgb = torch.swapaxes(image_rgb, -1, -2).cuda()

            # return image_rgb, image_depth
            return image_rgb

    # def callback(self, img_rgb,img_depth):
        # speed, steering = self.model(self.preprocess(img_rgb,img_depth)).detach().float().cpu().numpy().flatten()

    def steering_trans(self, steering):
        print('org-steering', steering)
        index_max = np.argmax(steering)
        print('the max index', index_max)
        steering = self.steering_class[index_max]

        return steering

    def callback(self, img, args):
        if args == "rgb":
            self.img_rgb = img
        elif args == "depth":
            self.img_depth = img

        # if self.img_rgb is not None and self.img_depth is not None:
        if self.img_rgb is not None:
            img_rgb = self.img_rgb
            # img_depth = self.img_depth
            # input_1, input_2 = self.preprocess(img_rgb, img_depth)
            # speed, steering = self.model(input_1, input_2)
            input_1 = self.preprocess(img_rgb)
            steering = self.model(input_1)
            # print('steering from the model',steering)
            # steering = self.steering_trans(np.array(steering))
            # speed *= -1
            # speed = speed.detach().float().cpu().numpy().flatten()
            steering = steering.detach().float().cpu().numpy().flatten()

        # print('steering : ',steering)
        # print('speed : ',speed)

        steering = self.steering_trans(np.array(steering))
        print('steering : ', steering)
        commande = AckermannDriveStamped()
        # commande.drive.speed = speed * self.speed_factor
        commande.drive.speed = float(1)
        commande.drive.steering_angle = steering
        self.pub.publish(commande)
        self.rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('FollowRoad_node', anonymous=True)
        p = follow_road()
    except rospy.ROSInterruptException:
        pass
