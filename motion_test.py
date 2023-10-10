
import URBasic
import math
import numpy as np

import time

import math3d as m3d


ROBOT_IP = '192.168.0.25'
#ACCELERATION = 0.9  # Robot acceleration value
#VELOCITY = 0.8  # Robot speed value

ACCELERATION = 0.4  # Robot acceleration value
VELOCITY = 0.4  # Robot speed value

robot_startposition = (math.radians(90),
                    math.radians(-90),
                    math.radians(90),
                    math.radians(0),
                    math.radians(90),
                    math.radians(0))

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)
print("moving to start position")
print(robot_startposition)
robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY )

robot.movej(pose=[ 0.0, -0.6609,  0.6255,    1.5694,  0.0117,  0.0147],a = ACCELERATION, v=VELOCITY)
# print("Get actual pose")
# position = robot.get_actual_tcp_pose()

#print(position)

# trans = tuple[0.001,0,0,0,0,0]
# print(trans)

#newPose = np.append(robot_startposition, trans)
#print(newPose)
#robot.movej(q=newPose, a= ACCELERATION, v= VELOCITY )

#robot.set_realtime_pose()
print("closing robot")

robot.close()
