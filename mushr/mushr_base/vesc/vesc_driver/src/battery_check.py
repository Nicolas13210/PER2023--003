#!/usr/bin/env python3

import rospy
from vesc_msgs.msg import VescStateStamped
import os

class Battery_check:
	def __init__(self):

		rospy.sleep(1)
		rospy.Subscriber("sensors/core",VescStateStamped,self.callback)
		print("battery check initialized")

	def callback(self,data):
		#rospy.loginfo(rospy.get_caller_id() + "Tension = " , data.state.voltage_input)
		tension=data.state.voltage_input
		print("Tension : " + str(tension))

		if tension < 14.3:
			print('Baterie faible, tension = ' + str(tension))
		elif tension <= 14:
			print("on se casse")
			os.system("rosnode kill -a")
		rospy.sleep(10)

if __name__ == '__main__':
	rospy.init_node('batt_listener', anonymous=True)
	batt=Battery_check()
	rospy.spin()
