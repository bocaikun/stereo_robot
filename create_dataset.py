from re import S
import pybullet as p
import pybullet_data
import numpy as np
import math
from PIL import Image
import datetime, time, json, os, csv

# define logs path and creat logs dir
data_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
data_index = 0 # defualt
train_left_image_path = 'data/original/%s/train/%s/image/left_images/%s.png'%(data_id,'%s','%s')
train_right_image_path = 'data/original/%s/train/%s/image/right_images/%s.png'%(data_id,'%s', '%s')
train_csv_path = 'data/original/%s/train/%s/csv/train_l.csv'%(data_id,'%s')

test_left_image_path = 'data/original/%s/test/%s/image/left_images/%s.png'%(data_id,'%s','%s')
test_right_image_path = 'data/original/%s/test/%s/image/right_images/%s.png'%(data_id,'%s','%s')
test_csv_path = 'data/original/%s/test/%s/csv/test_l.csv'%(data_id,'%s')


def creat_train_dir(data_id, data_index):
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/original'):
        os.makedirs('data/original/')
    if not os.path.exists('data/original/%s'%data_id):
        os.makedirs('data/original/%s'%data_id)
    if not os.path.exists('data/original/%s/train'%data_id):
        os.makedirs('data/original/%s/train'%data_id)
    if not os.path.exists('data/original/%s/train/%s'%(data_id, data_index)):
        os.makedirs('data/original/%s/train/%s'%(data_id,data_index))
    if not os.path.exists('data/original/%s/train/%s/csv'%(data_id, data_index)):
        os.makedirs('data/original/%s/train/%s/csv'%(data_id,data_index))
    if not os.path.exists('data/original/%s/train/%s/image/left_images'%(data_id, data_index)):
        os.makedirs('data/original/%s/train/%s/image/left_images'%(data_id, data_index))
    if not os.path.exists('data/original/%s/train/%s/image/right_images'%(data_id, data_index)):
        os.makedirs('data/original/%s/train/%s/image/right_images'%(data_id, data_index))

def creat_test_dir(data_id, data_index):
    if not os.path.exists('data/original/%s/test'%data_id):
        os.makedirs('data/original/%s/test'%data_id)
    if not os.path.exists('data/original/%s/test/%s'%(data_id, data_index)):
        os.makedirs('data/original/%s/test/%s'%(data_id,data_index))
    if not os.path.exists('data/original/%s/test/%s/csv'%(data_id, data_index)):
        os.makedirs('data/original/%s/test/%s/csv'%(data_id,data_index))
    if not os.path.exists('data/original/%s/test/%s/image/left_images'%(data_id, data_index)):
        os.makedirs('data/original/%s/test/%s/image/left_images'%(data_id, data_index))
    if not os.path.exists('data/original/%s/test/%s/image/right_images'%(data_id, data_index)):
        os.makedirs('data/original/%s/test/%s/image/right_images'%(data_id, data_index))

# camera setting
def image_output():
    # left camera position
    view_matrix1=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,-0.05,1.5],
                                                    distance=.1,
                                                    yaw=-90,
                                                    pitch=-60,
                                                    roll=0,upAxisIndex=2)
    # left camera parameter 
    proj_matrix1=p.computeProjectionMatrixFOV(fov=69,
                                            aspect=1.0,
                                            nearVal=0.1,
                                            farVal=100.0)
    # left camera image ouput
    (_,_,px1,_,_)=p.getCameraImage(width=200,height=200,
                                viewMatrix=view_matrix1,
                                projectionMatrix=proj_matrix1,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_array1=np.array(px1,dtype=np.uint8)
    rgb_array1=np.reshape(rgb_array1,(200,200,4))
    rgb_array1=rgb_array1[:,:,:3]
    rgb_array1 = Image.fromarray(rgb_array1)
    # right camera position
    view_matrix2=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,0.05,1.5],
                                                    distance=.1,
                                                    yaw=-90,
                                                    pitch=-60,
                                                    roll=0,upAxisIndex=2)
    # right camera parameter
    proj_matrix2=p.computeProjectionMatrixFOV(fov=69,aspect=1.0,
                                            nearVal=0.1,
                                            farVal=100.0)
    # right camera image output
    (_,_,px2,_,_)=p.getCameraImage(width=200,height=200,
                                viewMatrix=view_matrix2,
                                projectionMatrix=proj_matrix2,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
                                
    rgb_array2=np.array(px2,dtype=np.uint8)
    rgb_array2=np.reshape(rgb_array2,(200,200,4))
    rgb_array2=rgb_array2[:,:,:3]
    rgb_array2 = Image.fromarray(rgb_array2)
    return rgb_array1, rgb_array2

def calculate_xy(theta, b, a, r=0.15):
    theta = (theta / 3.141593) * 180 #()
    #y = b - abs(math.sin(math.radians(theta+270)))*r
    y = b - abs(math.cos(math.radians(theta)))*r
    x = a - math.sin(math.radians(theta))*r

    return x*1.333, y

# task setting
def set_env(x=0., y=0.9, rz=0., step=50, data_index=0, mode=0):    
    robotStartPos = [0.0000, 0.0000, 0.6500]
    robotStartOrn = p.getQuaternionFromEuler([0,0,0])
    p.resetBasePositionAndOrientation(robotId[0],robotStartPos,robotStartOrn)

    tableStartPos = [0.8000, 0., -0.0500]
    tableStartOrn = p.getQuaternionFromEuler([0,0,0])
    p.resetBasePositionAndOrientation(tableId,tableStartPos,tableStartOrn)

    boxStartPos = [y, x, 0.5800]
    boxStartOrn = [0.0000, 0.0000, rz]
    p.resetBasePositionAndOrientation(boxId,boxStartPos,p.getQuaternionFromEuler(boxStartOrn))

    # calculate move plane
    p.getNumJoints(robotId[0]) # get joint num
    p.getJointInfo(robotId[0], 6) # get joint inf
    robotStartPos = [0.6500, 0.0000, 0.6500]
    robotStartOrn = [1.5707965, 3.141593, 1.5707965]
    objX, objY = calculate_xy(boxStartOrn[2], boxStartPos[0], boxStartPos[1])
    # robotEndPos = [(boxStartPos[0]-robotStartPos[0])*ratio+robotStartPos[0], (boxStartPos[1]-robotStartPos[1])*ratio+robotStartPos[1], 0.625]
    robotEndPos = [objY, objX, 0.6500]
    robotEndOrn = [1.5707965, 3.141593, boxStartOrn[2]+1.5707965]

    startPos_array = np.array(robotStartPos)
    endPos_array = np.array(robotEndPos)
    startOrn_array = np.array(robotStartOrn)
    endOrn_array = np.array(robotEndOrn)

    targetPositionsJoints = p.calculateInverseKinematics(robotId[0], 6, startPos_array,\
        targetOrientation=p.getQuaternionFromEuler(startOrn_array))# IK
    p.setJointMotorControlArray(robotId[0], range(7), p.POSITION_CONTROL,\
            targetPositions=targetPositionsJoints) # move plan

    stepPos_array = (endPos_array - startPos_array) / step
    stepOrn_array = (endOrn_array - startOrn_array) / step

    # Initializing
    print("Initializing")
    if mode == 0:
        creat_train_dir(data_id, data_index)
    if mode == 1:    
        creat_test_dir(data_id, data_index)

    for i in range(step):
        #print(i, "step")
        robotStepPos = list(startPos_array) # next position
        robotStepOrn = list(startOrn_array)
        targetPositionsJoints = p.calculateInverseKinematics(robotId[0], 6, robotStepPos,\
            targetOrientation=p.getQuaternionFromEuler(robotStepOrn))# IK
        p.setJointMotorControlArray(robotId[0], range(7), p.POSITION_CONTROL,\
            targetPositions=targetPositionsJoints) # move plan
        # for i in range(10): #time
        p.stepSimulation()
        #time.sleep(1/240)

    # start data collection
    print("Start collecting data")
    robot_rows = []
    for i in range(step+1):
        if mode == 0:
            left_img_dir = train_left_image_path%(data_index,str(i).zfill(3))
            right_img_dir = train_right_image_path%(data_index,str(i).zfill(3))
        if mode == 1:
            left_img_dir = test_left_image_path%(data_index,str(i).zfill(3))
            right_img_dir = test_right_image_path%(data_index,str(i).zfill(3))
        #print(i, "step")
        robotStepPos = list(np.round((stepPos_array*i + startPos_array),8))
        robotStepOrn = list(np.round((stepOrn_array*i + startOrn_array),8))
        robot_rows.append(robotStepPos + robotStepOrn)
        #print("step:",i, "getL:", robot_rows[i])
        targetPositionsJoints = p.calculateInverseKinematics(robotId[0], 6, robotStepPos,\
            targetOrientation=p.getQuaternionFromEuler(robotStepOrn))# IK
        p.setJointMotorControlArray(robotId[0], range(7), p.POSITION_CONTROL,\
                targetPositions=targetPositionsJoints) # move plan
        # for i in range(10): #time
        left_img, right_img = image_output()
        left_img.save(left_img_dir)
        right_img.save(right_img_dir)
        p.stepSimulation()
        #time.sleep(1/120)
    if mode == 0:
        csvname = train_csv_path%(data_index)
    if mode == 1:
        csvname = test_csv_path%(data_index)
    f = open(csvname,'w', newline='',encoding='utf-8')
    f_csv = csv.writer(f)
    f_csv.writerows(robot_rows)
    p.resetSimulation()
    print("Index:",data_index," Data collection done")

if __name__ == "__main__":
    # pybullet env setting
    #p.connect(p.GUI) # GUI Window
    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # load pybullet data path
    p.setGravity(0, 0, 0) # set gravity

    # dataset setting
    traindataset_num = 3 * 50
    testdataset_num = 2 * 10
    total_num = traindataset_num + testdataset_num
    train_x = [0.3, 0.0, -0.3]
    test_x = [0.15, -0.15]

    # data collection
    for i in range(total_num):
        # load object and robot urdf
        planeID = p.loadURDF("plane.urdf")
        robotId = p.loadSDF("kuka_iiwa/model.sdf")
        tableId = p.loadURDF("table/table.urdf")
        boxId = p.loadURDF("objects/mug.urdf", globalScaling=1.2)
        #boxId = p.loadURDF("duck_vhacd.urdf", globalScaling=2)

        # switch mode
        y = 0.9 + np.random.uniform(-1, 1) * 0.01
        rz = np.random.uniform(-1, 1) * 0.785398 
        if i < traindataset_num:
            print("Train Data: ", i)
            x = np.random.choice(train_x,1).squeeze() + np.random.uniform(-1, 1) * 0.01
            set_env(x,y,rz,data_index=i,mode=0)
        else:
            print("Test Data: ", i-traindataset_num)
            x = np.random.choice(test_x,1).squeeze() + np.random.uniform(-1, 1) * 0.01
            set_env(x,y,rz,data_index=i-traindataset_num,mode=1)

    p.disconnect()
    print("All data collection done")