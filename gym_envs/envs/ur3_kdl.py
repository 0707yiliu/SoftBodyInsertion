# MODEL: Univewrsal Robots UR3 with PyKDL+pykdl_utils
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import math
import numpy as np
import PyKDL as kdl

from kdl_parser.urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
# from kdl_parser.urdf_parser_py.kdl_parserTree import urdf_tree # it contains the kdl_tree_from_urdf_model
from scipy.spatial.transform import Rotation as R
from gym_envs.utils import euler_to_quaternion

d2r = math.pi / 180

class URx_kdl():
    def __init__(self, DHfile) -> None:
        robot = URDF.from_xml_file(DHfile)
        tree = kdl_tree_from_urdf_model(robot)
        
        self.chain = tree.getChain("base_link", "tool0")
        # print("the UR3 .urdf model has %d bodies." % tree.getNrOfSegments())
        # print("the UR3 has %d bodies we used to controlled" % chain.getNrOfSegments())
        # print("the UR3 has %d joints we controlled" % chain.getNrOfJoints())

    def forward(self, qpos):
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        pos = kdl.Frame()
        q = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q[i] = qpos[i]
        fk_flag = fk.JntToCart(q, pos)
        f_pos = np.zeros(3)
        for i in range(3):
            f_pos[i] = pos.p[i]
        return f_pos
    
    def inverse(self, init_joint, goal_pose, goal_rot):
        try:
            rot = kdl.Rotation()
            rot = rot.Quaternion(goal_rot[0], goal_rot[1], goal_rot[2], goal_rot[3]) # radium x y z w
            pos = kdl.Vector(goal_pose[0], goal_pose[1], goal_pose[2])
        except ValueError:
            print("The target pos can not be transfor to IK-function.")
        target_pos = kdl.Frame(rot, pos)
        # print(target_pos)
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        #inverse kinematics
        ik_v = kdl.ChainIkSolverVel_pinv(self.chain)
        # ik = kdl.ChainIkSolverPos_NR(chain, fk, ik_v, maxiter=100, eps=math.pow(10, -9))

        # try:
        #     q_min = kdl.JntArray(len(joint_limit_lower))
        #     q_max = kdl.JntArray(len(joint_limit_lower))
        #     for i in range(len(joint_limit_lower)):
        #         q_min[i] = joint_limit_lower[i]
        #         q_max[i] = joint_limit_lower[i]
        # except ValueError:
        #     print("you should input the joint limitation value.")

        # ik_p_kdl = kdl.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk, ik_v)
        ik_p_kdl = kdl.ChainIkSolverPos_NR(self.chain, fk, ik_v)
        q_init = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(6):
            q_init[i] = init_joint[i]
        q_out = kdl.JntArray(self.chain.getNrOfJoints())
        ik_p_kdl.CartToJnt(q_init, target_pos, q_out)
        # print("Output angles:", q_out)
        q_out_trans = np.zeros(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q_out_trans[i] = np.array(q_out[i])
        # print(q_out_trans)
        return (q_out_trans)

# # robot = URDF.from_xml_file("/home/yi/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot.urdf")
# robot = URDF.from_xml_file("/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot_real.urdf")
# tree = kdl_tree_from_urdf_model(robot)


# chain = tree.getChain("base_link", "tool0")
# # print("the UR3 .urdf model has %d bodies." % tree.getNrOfSegments())
# # print("the UR3 has %d bodies we used to controlled" % chain.getNrOfSegments())
# # print("the UR3 has %d joints we controlled" % chain.getNrOfJoints())

# d2r = math.pi / 180

# def forward(qpos):
#     #forward kinematics
#     fk = kdl.ChainFkSolverPos_recursive(chain)
#     pos = kdl.Frame()
#     q = kdl.JntArray(chain.getNrOfJoints())
#     for i in range(chain.getNrOfJoints()):
#         q[i] = qpos[i]
#     # 23.7168     9.97208      1.0109    -19.4201    -25.1327     5.60529
#     # 31.7856     4.45017     1.51662    -40.5243    -33.2015     38.0089
#     # print("fk q", q)
#     fk_flag = fk.JntToCart(q, pos)
#     f_pos = np.zeros(3)
#     for i in range(3):
#         f_pos[i] = pos.p[i]
#     # print("fk_flag", fk_flag)
#     # print("pos", pos)
#     # print("rotation matrix of ee", pos.M)
#     # print("urdf end-effector position:", pos.p)
#     # print("urdf end-effector quaternion:",kdl.Rotation(pos.M).GetQuaternion())
#     return f_pos
    
    
# def inverse(init_joint, goal_pose, goal_rot):
#     try:
#         rot = kdl.Rotation()
#         rot = rot.Quaternion(goal_rot[0], goal_rot[1], goal_rot[2], goal_rot[3]) # radium x y z w
#         pos = kdl.Vector(goal_pose[0], goal_pose[1], goal_pose[2])
#     except ValueError:
#         print("The target pos can not be transfor to IK-function.")
#     target_pos = kdl.Frame(rot, pos)
#     # print(target_pos)
#     fk = kdl.ChainFkSolverPos_recursive(chain)
#     #inverse kinematics
#     ik_v = kdl.ChainIkSolverVel_pinv(chain)
#     # ik = kdl.ChainIkSolverPos_NR(chain, fk, ik_v, maxiter=100, eps=math.pow(10, -9))

#     # try:
#     #     q_min = kdl.JntArray(len(joint_limit_lower))
#     #     q_max = kdl.JntArray(len(joint_limit_lower))
#     #     for i in range(len(joint_limit_lower)):
#     #         q_min[i] = joint_limit_lower[i]
#     #         q_max[i] = joint_limit_lower[i]
#     # except ValueError:
#     #     print("you should input the joint limitation value.")

#     # ik_p_kdl = kdl.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk, ik_v)
#     ik_p_kdl = kdl.ChainIkSolverPos_NR(chain, fk, ik_v)
#     q_init = kdl.JntArray(chain.getNrOfJoints())
#     for i in range(6):
#         q_init[i] = init_joint[i]
#     q_out = kdl.JntArray(chain.getNrOfJoints())
#     ik_p_kdl.CartToJnt(q_init, target_pos, q_out)
#     # print("Output angles:", q_out)
#     q_out_trans = np.zeros(chain.getNrOfJoints())
#     for i in range(chain.getNrOfJoints()):
#         q_out_trans[i] = np.array(q_out[i])
#     # print(q_out_trans)
#     return (q_out_trans)


# # if __name__ == "__main__":
# #     qpos = [ 1.36480364, -1.56298705,  1.30087565, -1.30868492, -1.57079633 ,-0.20599269]
# #     eepos = [ -0.05, 0.31, 1.05]
# #     goal_rot = euler_to_quaternion(180 * d2r, 0, 60 * d2r)
# #     goal_rot = np.roll(goal_rot, 3)
# #     for i in range(3):
# #         print("inv:", inverse(qpos, eepos, goal_rot))
# #         print(forward(inverse(qpos, eepos, goal_rot)))
