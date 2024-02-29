#!/usr/bin/env python3

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#import tensorflow as tf
#import scipy.misc
#import model
#from subprocess import call

class follow_road:
    def __init__(self):
        print('init')
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 1)
        #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/test_03_02_2023_preprocessedimg.pth'))
        self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/40mn_datasettest_23-02-2023_18-12.pth'))
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.model = self.model.eval().half()

	# Pre-porcessing image function
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

        self.steering_gain=float(rospy.get_param("steering_gain"))
        self.steering_dgain=float(rospy.get_param("steering_dgain"))
        self.steering_bias=float(rospy.get_param("steering_bias"))
        self.speed=float(rospy.get_param("speed"))

        self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=10)
        self.rate=rospy.Rate(10)
        
        self.bridge = CvBridge()
        self.angle_last=0.0
	
        rospy.Subscriber("/car/camera/color/image_raw", Image, self.callback)

        self._run()

    def _run(self):
        rospy.spin()

    def preprocess(self, img):
        image= self.bridge.imgmsg_to_cv2(img, "bgr8")
        #cv2.imwrite('/root/mushr00SD/imgtest.png',image)
        height , width = image.shape[:2]
        image=image[int(2*height/3):height,:,:]
        img_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,img_thresh = cv2.threshold(img_grey,170,255,cv2.THRESH_BINARY)
        image=cv2.Canny(img_thresh,50,150)
        #cv2.imwrite('/root/mushr00SD/rettest.png',image)
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #cv2.imwrite('/root/mushr00SD/test.png',image)
        image = PIL.Image.fromarray(image)
        #image.save('/root/mushr00SD/testPIL.png')
        '''image= self.bridge.imgmsg_to_cv2(img, "rgb8")
        image = PIL.Image.fromarray(image)
        width, height = image.size
        image=image.crop((0,2*height/3,width,height))
        print(image.size)'''
        image = transforms.functional.resize(image, (120,640))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def callback(self, img):
        steering = self.model(self.preprocess(img)).detach().float().cpu().numpy().flatten()
        print('steering : ',steering)
        commande=AckermannDriveStamped()
        commande.drive.speed=self.speed
        commande.drive.steering_angle=steering*self.steering_gain
        self.pub.publish(commande)
        self.rate.sleep()


if __name__=="__main__":
    try:
        rospy.init_node('FollowRoad_node', anonymous=True)
        p = follow_road()
    except rospy.ROSInterruptException:
        pass

