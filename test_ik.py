# import numpy as np
# from numpy import pi as PI
# import sys

# import PyKDL
# from PyKDL import *



# fetch_chain = Chain()
# fetch_chain.addSegment(Segment(Joint(Joint.Fixed), Frame(Rotation.RPY(0, 0, 0), Vector(0.0, 0.0, 0.0))))
# fetch_chain.addSegment(Segment(Joint(Joint.Fixed), Frame(Rotation.RPY(0, 0, 0), Vector(0, -0.19, 0.3))))
# fetch_chain.addSegment(Segment(Joint(Joint.Fixed), Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotY), Frame(Rotation.RPY(0, 0, 0), Vector(0.0, 0.0, 0.0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotZ), Frame(Rotation.RPY(0, 0, 0), Vector(0.0, 0.0, 0.0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotX), Frame(Rotation.RPY(0, 0, 0), Vector(0.19, 0, 0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotY), Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotX), Frame(Rotation.RPY(0, 0, 0), Vector(0.28, 0, 0))))
# fetch_chain.addSegment(Segment(Joint(Joint.RotY), Frame(Rotation.RPY(0, 0, 0), Vector(0.06, 0.03, 0))))
# fetch_chain.addSegment(Segment(Joint(Joint.Fixed), Frame(Rotation.RPY(0, 0, 0), Vector(0.05, 0.0, 0.0))))

# print("use fk and ik in module derectly ! ")

# q0 = JntArray(6)
# q0_list = [0.7, -0.4, -1.5, -1.4, 1.0, -0.2]
# for i in range(6):
# 	q0[i] = q0_list[i]
# p0 = Frame()

# # JntToCart(fetch_chain, q0, p0)
# # print(p0.p[0], p0.p[1], p0.p[2])

# # q1 = JntArray(6)
# # CartToJnt(fetch_chain, q0, p0, q1)
# # for i in range(6):
# # 	print(q1[i])




# print("use fk and ik in solver ! ")

# jac_solver = ChainJntToJacSolver(fetch_chain)

# ik_v_kdl = ChainIkSolverVel_pinv(fetch_chain)
# fk_kdl = ChainFkSolverPos_recursive(fetch_chain)

# q_min = JntArray(6)
# q_max = JntArray(6)
# for i in range(6):
#     q_min[i] = -3
#     q_max[i] = 3


# ik_p_kdl = ChainIkSolverPos_NR_JL(fetch_chain, q_min, q_max, fk_kdl, ik_v_kdl)


# p0 = Frame()
# q1 = JntArray(6)
# fk_kdl.JntToCart(q0, p0, -1)


# p1 = Frame(Rotation.RPY(-0.05, 0.1, 1.1), Vector(0.3, -0.05, 0.07))
# ik_p_kdl.CartToJnt(q0, p0, q1)
# for i in range(6):
# 	print(q1[i])

import math
import numpy as np
import PyKDL as kdl

from kdl_parser.urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
# from kdl_parser.urdf_parser_py.kdl_parserTree import urdf_tree # it contains the kdl_tree_from_urdf_model


robot = URDF.from_xml_file("gym_envs/models/ur3_robot.urdf")
tree = kdl_tree_from_urdf_model(robot)

print(tree.getNrOfSegments())
chain = tree.getChain("base_link", "ee_link")
print(chain.getNrOfSegments())
print(chain.getNrOfJoints())

#forward kinematics
fk = kdl.ChainFkSolverPos_recursive(chain)

pos = kdl.Frame()
q = kdl.JntArray(6)
for i in range(6):
    q[i] = 0
q[0] = 31.7856 
q[1] = 4.45017
q[2] = 1.51662
q[3] = -40.5243
q[4] = -33.2015
q[5] = 38.0089
# 23.7168     9.97208      1.0109    -19.4201    -25.1327     5.60529
# 31.7856     4.45017     1.51662    -40.5243    -33.2015     38.0089
print("fk q", q)
fk_flag = fk.JntToCart(q, pos)
# print("fk_flag", fk_flag)
# print("pos", pos)
print("rotation matrix of ee", pos.M)
print("end-effector position:", pos.p)

#inverse kinematics
ik_v = kdl.ChainIkSolverVel_pinv(chain)
ik = kdl.ChainIkSolverPos_NR(chain, fk, ik_v, maxiter=100, eps=math.pow(10, -9))

q_init = kdl.JntArray(6)
q_out = kdl.JntArray(6)
ik.CartToJnt(q_init, pos, q_out)
print("Output angles (rads):", q_out)