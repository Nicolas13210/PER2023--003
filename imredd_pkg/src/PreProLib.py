'''
This script saves each topic in a bagfile as a csv.
Accepts a filename as an optional argument. Operates on all bagfiles in current directory if no argument provided
Usage1 (for one bag file):
	python bag2csv.py filename.bag
Usage 2 (for all bag files in current directory):
	python bag2csv.py
Written by Nick Speal in May 2013 at McGill University's Aerospace Mechatronics Laboratory. Bugfixed by Marc Hanheide June 2016.
www.speal.ca
Supervised by Professor Inna Sharf, Professor Meyer Nahon
'''

import rosbag, sys, csv
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import glob
import pandas as pd
import numpy as np
from PIL import Image

# if (len(sys.argv) > 2):
#     print("invalid number of arguments:   " + str(len(sys.argv)))
#     print("should be 2: 'bag2csv.py' and 'bagName'")
#     print("or just 1  : 'bag2csv.py'")
#     sys.exit(1)
# elif (len(sys.argv) == 2):
#     listOfBagFiles = [sys.argv[1]]
#     numberOfFiles = 1
#     print("reading only 1 bagfile: " + str(listOfBagFiles[0]))
# elif (len(sys.argv) == 1):
#     listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
#     numberOfFiles = str(len(listOfBagFiles))
#     print("reading all " + numberOfFiles + " bagfiles in current directory: \n")
#     for f in listOfBagFiles:
#         print(f)
#     print("\n press ctrl+c in the next 10 seconds to cancel \n")
#     time.sleep(10)
# else:
#     print("bad argument(s): " + str(sys.argv))	#shouldnt really come up
#     sys.exit(1)

def bag2csv(path):
    listOfBagFiles = []
    for file in os.listdir(path):
        if file[-4:] == '.bag' :
            listOfBagFiles.append(file)
    numberOfFiles = len(listOfBagFiles)

    count = 0
    for bagFile in listOfBagFiles:
        count += 1
        print("reading file " + str(count) + " of  " + str(numberOfFiles) + ": " + bagFile)
        #access bag
        bag = rosbag.Bag(path+'/'+bagFile)
        bagContents = bag.read_messages()
        bagName = bagFile

        #create a new directory
        folder = path+'/'+bagName.rstrip(".bag")
        try:	#else already exists
            os.makedirs(folder)
        except:
            pass
        shutil.copyfile(path+'/'+bagName, folder + '/' + bagName)


        #get list of topics from the bag
        listOfTopics = []
        for topic, msg, t in bagContents:
            if topic not in listOfTopics:
                listOfTopics.append(topic)


        for topicName in listOfTopics:
            #Create a new CSV file for each topic
            filename = folder + '/' + topicName.replace('/', '_slash_') + '.csv'
            with open(filename, 'w+') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                firstIteration = True	#allows header row
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                    #	- put it in the form of a list of 2-element lists
                    msgString = str(msg)
                    msgList = msgString.split('\n')
                    instantaneousListOfData = []
                    for nameValuePair in msgList:
                        splitPair = nameValuePair.split(':')
                        for i in range(len(splitPair)):	#should be 0 to 1
                            splitPair[i] = splitPair[i].strip()
                        instantaneousListOfData.append(splitPair)
                    #write the first row from the first element of each pair
                    if firstIteration:	# header
                        headers = ["rosbagTimestamp"]	#first column header
                        for pair in instantaneousListOfData:
                            headers.append(pair[0])
                        filewriter.writerow(headers)
                        firstIteration = False
                    # write the value from each pair to the file
                    values = [str(t)]	#first column will have rosbag timestamp
                    for pair in instantaneousListOfData:
                        if len(pair)>1:
                            values.append(pair[1])
                    filewriter.writerow(values)
        bag.close()
    print("Done reading all " + str(numberOfFiles) + " bag files.")

def csv2png(path):
    print("Converting vector images from CSV to png files...")
    # Cherche tous les csv du directory
    list_dir = [x[0] for x in os.walk(path)] # arbre des folders du path
    csv_path = []
    for k in range(len(list_dir)) :
        if list_dir[k].find('.') ==-1 : # cherche tous les folders sans . (pour pas rechercher sur les git)
            csv_path.append(glob.glob(os.path.join(list_dir[k], "*.csv"))) # cherche tous les csv dans le folder
    csv_path = sum(csv_path, []) # enlève le nesting de listes

    # ouvre csv avec image
    img = []
    img_path = []
    for k in range(len(csv_path)) :
        if csv_path[k][-13:-4] =="image_raw" :
            img.append(pd.read_csv(csv_path[k]))
            img_path.append(csv_path[k])

    #création nouveau folder
    try:	#else already exists
        os.makedirs(path+'/'+"images")
    except:
        pass

    #extraction image et time-stamp
    for k in range(len(img)) :
        height = img[k].height[0]
        width = img[k].width[0]
        n = len(img[k].data)
        t = img[k].rosbagTimestamp.astype("string")
        name = img_path[k].split('\\')[-2]

        #création ouveau folder
        try:	#else already exists
            os.makedirs(path+'/'+"images/"+name)
        except:
            pass

        # csv2png
        list_img = []
        time = []
        for l in range(len(img[0].data)) :
            if l % 100 == 0:
                print("Converted ",l,"images of",len(img[0].data))
            mat = np.fromstring(img[k].data[l][1:-1], dtype=int, sep=',') #remplace string de vecteur en np.array
            list_img.append(mat.reshape(height, width, 3)) #image vecteur en image matricielle
            time.append(int(t[l][:-6])) # suppression en dessous du milième de seconde
            im = Image.fromarray(list_img[l].astype(np.uint8), "RGB")
            im.save(path+"/images/"+name+"/img_"+name+str(time[l])+".png")
