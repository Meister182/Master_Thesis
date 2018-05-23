#! /usr/bin/env python
#===============IMPORTS=====================
import cv2
import copy
import rospy
import numpy as np
#===============ROS/mensagens===============
from cv_bridge import CvBridge                  #comunicacao entre ROS e Python
from sensor_msgs.msg import Image               #kinect
#from sensor_msgs.msg import PointCloud2        #kinect


#======================   Class Kinect   ====================== 
class Kinect_object(object):
    def __init__(self):
        #===================parametos da class=====================
        self.imgRGB = np.zeros((480, 640, 3))       	 #rgb image
        self.imgDepth = np.zeros((480, 640))    	 #depth image


        #=====================Metodos heradados======================
        self.bridge = CvBridge()        #convercao ROS->OpenCV

        #============Subscrever aos topicos de interesse==============
        rospy.init_node('get_kinect_info')  #inicia o nodo do ros, relacionado a este processo! apenas pode haver 1 nodo por processo...
        rospy.Subscriber( '/camera/rgb/image_color', Image, self.get_RGB)
        rospy.Subscriber( '/camera/depth_registered/image_raw', Image, self.get_Depth)


    #Metodos
    def get_RGB(self,image):
        self.imgRGB = self.bridge.imgmsg_to_cv2(image, "rgb8")
        
    def get_Depth(self,image):
        self.imgDepth = self.bridge.imgmsg_to_cv2(image)

    def get_RGBD(self):
        tmp_rgb = self.imgRGB
        tmp_dep = self.imgDepth
        tmp_dep = np.expand_dims(tmp_dep,axis=2)
        #rgbD
        RGBD = np.concatenate((tmp_rgb, tmp_dep),axis=2)

        return RGBD