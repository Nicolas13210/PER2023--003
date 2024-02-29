import rosbag
import subprocess
import yaml
import numpy as np
from PIL import Image
from config import cfg
import os 


def preprocess_gold_data():
    print('start preprocessing')
    bag = rosbag.Bag(cfg.DATASET.GOLD_DATA_PATH)
    print('bag open')

    instant_index = 0

    past_image = None
    past_depth = None
    past_lid = None
    past_command = None
    topic_count = 0
    for topic, msg, t in bag.read_messages():

        topic_count += 1
        if topic_count < 100:
            continue

        if past_image is not None and past_depth is not None and past_lid is not None and past_command is not None:
            min_list = [abs(past_lid[-1].to_sec() - past_image[-1].to_sec()), abs(past_lid[-1].to_sec() - past_depth[-1].to_sec()), abs(past_lid[-1].to_sec() - past_command[-1].to_sec())]
            if abs(t.to_sec() - past_lid[-1].to_sec()) > max(min_list):
                save_instant(past_image, past_depth, past_command, past_lid, instant_index)
                print(f'instant {instant_index} created')
                instant_index += 1

                past_image = None
                past_depth = None
                past_lid = None
                past_command = None

        if topic == '/car/scan':
            if past_lid is not None:
                print("ERROR !!!!!!!")
                return
            past_lid = (topic, msg, t)

        elif topic == '/car/camera/color/image_raw' :
            if past_image is None or past_lid is None or (abs(past_lid[-1].to_sec() - t.to_sec()) < abs(past_lid[-1].to_sec() - past_image[-1].to_sec())):
                past_image = (topic, msg, t)
        elif topic == '/car/camera/depth/image_rect_raw' :
            if past_depth is None or past_lid is None or (abs(past_lid[-1].to_sec() - t.to_sec()) < abs(past_lid[-1].to_sec() - past_depth[-1].to_sec())):
                past_depth = (topic, msg, t)
        else:
            if past_command is None or past_lid is None or (abs(past_lid[-1].to_sec() - t.to_sec()) < abs(past_lid[-1].to_sec() - past_command[-1].to_sec())):
                past_command = (topic, msg, t) 
    
    print("preprocess data ended")
    print(f'last index is : {instant_index}')

def save_instant(past_image, past_depth, past_command, past_lid, index):
    save_image(past_image[1], os.path.join(cfg.DATASET.IMAGE_FOLDER_PATH,f'image_instant_{index}.jpeg'))
    save_depth_image(past_depth[1], os.path.join(cfg.DATASET.DEPTH_FOLDER_PATH,f'depth_0_instant_{index}.jpeg'), os.path.join(cfg.DATASET.DEPTH_FOLDER_PATH,f'depth_1_instant_{index}.jpeg'))
    save_commands(past_command[1], index, cfg.DATASET.COMMAND_FILE_PATH)
    save_lidar(past_lid[1], index, cfg.DATASET.LIDAR_FILE_PATH)

def save_image(msg, image_name):

    # img_size = (240,424,3)
    
    msg_image = msg.data
    msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)
    # img_1 = np.random.rand(240,424,3)

    # for i in range(img_1.shape[0]):
    #     for j in range(img_1.shape[1]):
    #         for k in range(img_1.shape[2]):
    
    #             img_1[i][j][k] = msg_image[k+3*j+i*424*3]

    
    img_1 = np.reshape(msg_image_2, (240,424,3), order='C')
    
    im_1 = Image.fromarray(img_1.astype(np.uint8))
    im_1.save(image_name)

def save_depth_image(msg, image_name_chanel_1, image_name_chanel_2):

    # img_size = (480,848,2)
    
    msg_image = msg.data
    msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)
    # img_1 = np.random.rand(480,848,2)

    # for i in range(img_1.shape[0]):
    #     for j in range(img_1.shape[1]):
    #         for k in range(img_1.shape[2]):
    
    #             img_1[i][j][k] = msg_image[k+2*j+i*848*2]

    img_1 = np.reshape(msg_image_2, (480,848,2), order='C')

    
    im_1 = Image.fromarray(img_1[:,:,0].astype(np.uint8))
    im_1.save(image_name_chanel_1)

    im_2 = Image.fromarray(img_1[:,:,1].astype(np.uint8))
    im_2.save(image_name_chanel_2)



def save_commands(msg,instant,filepath):

    steering_angle_velocity = msg.drive.steering_angle_velocity
    steering_angle = msg.drive.steering_angle
    speed = msg.drive.speed
    acceleration = msg.drive.acceleration

    dico = {'instant': instant, 
            'steering_angle_velocity' : steering_angle_velocity,
            'steering_angle' : steering_angle,
            'speed': speed,
            'acceleration':acceleration}

    with open(filepath,"a+") as f: 
        f.write(str(dico)+"\n")

def save_lidar(msg,instant,filepath):

    ranges = list(msg.ranges)
    intensities = list(msg.intensities)
    
    dico = {'instant': instant, 
            'ranges' : ranges,
            'intensities' : intensities}

    with open(filepath,"a+") as f: 
        f.write(str(dico)+"\n")
    
       



if __name__ == '__main__':

    bag = rosbag.Bag('./../data/data_try1.bag')

    i = 0
    old_t = 0
    for topic, msg, t in bag.read_messages(topics = ['/car/scan']):
        # print(len(msg.ranges),type(msg.ranges))
        # print(len(msg.intensities),type(msg.intensities))
        
        if i%100 == 0:
            save_lidar(msg,i,'./../data/try_1/lidar.json')

        if i > 1000:
            break

        i += 1
        # print(f"topic : {topic} , time_stamp : {t} , time_difference : {t.to_sec()-old_t}")
        # print('***********************************')

        # old_t=t.to_sec()

        # if i > 1000:
        #     break

    bag.close()
