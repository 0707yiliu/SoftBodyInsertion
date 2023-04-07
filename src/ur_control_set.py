# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 23/01/2023
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering
# UR-control SET

# import rtde_control
# HOST = "10.42.0.163"
# rtde_c = rtde_control.RTDEControlInterface(HOST)

# # Parameters
# velocity = 0.1
# acceleration = 0.1
# dt = 1.0/500  # 2ms
# lookahead_time = 0.1
# gain = 300
# joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]

# # Move to initial joint position with a regular moveJ
# rtde_c.moveJ(joint_q)

# # Execute 500Hz control loop for 2 seconds, each cycle is 2ms
# for i in range(1000):
#     t_start = rtde_c.initPeriod()
#     rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
#     joint_q[0] += 0.001
#     joint_q[1] += 0.001
#     rtde_c.waitPeriod(t_start)

# rtde_c.servoStop()
# rtde_c.stopScript()


import socket
import time
import struct
import numpy as np
import rtde_control
import rtde_receive
import math
import robotiq_gripper 

HOST = "10.42.0.163"
PORT = 30002

class UR:
    def __init__(self) -> None:
        # tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # tcp_client.settimeout(10)
        # tcp_client.connect((HOST, PORT))
        # self.tcp_client = tcp_client
        # self.back_log = 5
        # self.buffer_size = 1060
        self.rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
        self.rtde_c = rtde_control.RTDEControlInterface(HOST)
        actual_q = np.array(self.rtde_r.getActualQ())
        print(actual_q)
# [-1.54005605 -1.32302289  1.5954693  -1.86093154 -1.55838472 -5.41929955]
# [-0.71953518 -1.3279307   1.53478462 -1.75492777 -1.58143217 -5.10484416]

    def ee_ft_test(self):
        while True:
            ee_pose = self.rtde_r.getActualTCPPose()
            ft = self.rtde_r.getActualTCPForce()
            actual_q = np.array(self.rtde_r.getActualQ())
            ee_vel = self.rtde_r.getActualTCPSpeed()
            # j_q1 = [-1.54005605 , -1.32302289,  1.5954693,  -1.86093154, -1.55838472, -5.41929955]
            # j_q2 = [-0.71953518 , -1.32302289,  1.5954693,  -1.86093154, -1.55838472, -5.41929955]
            # if (i % 2) == 0:
            #     j_q = j_q1
            # else:
            #     j_q = j_q2
            # self.rtde_c.moveJ(j_q, 0.2, 0.2, True)
            print("ft:",ft)
            # print("ee pos:",np.array(ee_pose)[:3])
            # print("q pos", actual_q)
            # print("ee vel:", ee_vel)
            print("--------------------")
            # time.sleep(0.1)

    def control_test(self):
        vel = 0.01
        acc = 0.01
        dt = 1.0 / 300
        lookahead_time = 0.1
        gain = 2000
        j_q = [-1.54005605, -1.32302289,  1.5954693,  -1.86093154, -1.55838472, -5.41929955]
        self.rtde_c.moveJ(j_q, 0.2, 0.2)
        for i in range(1000):
            t_start = self.rtde_c.initPeriod()
            # print(t_start)
            self.rtde_c.servoJ(j_q, vel, acc, dt, lookahead_time, gain)
            j_q[0] += 0.00015
            j_q[1] -= 0.00015
            # time.sleep(0.5)
            ee_vel = self.rtde_r.getActualTCPSpeed()
            print("ee vel:", ee_vel)
            self.rtde_c.waitPeriod(t_start)
        time.sleep(3)
        # self.rtde_c.servoStop()
        # self.rtde_c.stopScript()

    def getEECurrentPos(self):
        try:
            packet_1 = self.tcp_client.recv(4)
            packet_2 = self.tcp_client.recv(8)
            packet_3 = self.tcp_client.recv(48)
            packet_4 = self.tcp_client.recv(48)
            packet_5 = self.tcp_client.recv(48)
            packet_6 = self.tcp_client.recv(48)
            packet_7 = self.tcp_client.recv(48) 
            packet_8 = self.tcp_client.recv(48)
            packet_9 = self.tcp_client.recv(48)
            packet_10 = self.tcp_client.recv(48)
            packet_11 = self.tcp_client.recv(48)
            
            packet_12 = self.tcp_client.recv(8)
            x = struct.unpack("!d", packet_12)[0]
            
            packet_13 = self.tcp_client.recv(8)
            y = struct.unpack("!d", packet_13)[0]

            packet_14 = self.tcp_client.recv(8)
            z = struct.unpack("!d", packet_14)[0]

            packet_15 = self.tcp_client.recv(8)
            rx = struct.unpack("!d", packet_15)[0]
            
            packet_16 = self.tcp_client.recv(8)
            ry = struct.unpack("!d", packet_16)[0]

            packet_17 = self.tcp_client.recv(8)
            rz = struct.unpack("!d", packet_17)[0]
            print(rz)

            

        except socket.error as socketerror:
            print("Error:", socketerror)
            print("Unable to read current position.")

ur3e = UR()
# ur3e.getEECurrentPos()
# ur3e.control_test()
# ur3e.ee_ft_test()

def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")

print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(HOST, 63352)
print("Activating gripper...")
gripper.activate()

print("Testing gripper...")
gripper.move_and_wait_for_pos(255, 100, 100)
log_info(gripper)
gripper.move_and_wait_for_pos(0, 255, 255)
log_info(gripper)
for i in range(20):
    gripper.move_and_wait_for_pos(i*10, 255, 255)
    time.sleep(0.5)
    log_info(gripper)