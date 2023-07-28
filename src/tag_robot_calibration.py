import cv2
import apriltag
import numpy as np
import matplotlib.pyplot as plt
import sys, os, math, time

import rtde_control
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl

import rtde_receive

from signal import signal, SIGINT

# urdf ik fk test ------------
import gym_envs.envs.ur3_kdl as urkdl
# urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur5e_gripper/ur5e_gripper_real.urdf"
urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot_real.urdf"
urxkdl = urkdl.URx_kdl(urDH_file)
# target1joint = np.array([1.57, -1.57, 1.57,  -1.57, -1.57,  0.])
target1joint = np.array([2.96539662, -1.57342686,  1.3205609,  -1.31793037, -1.57079633,  -1.7485])
target1joint = np.array([3.14, -1.57, 1.57, -1.57, -1.57, -1.57])
target1joint = np.array([2.96539662, -1.57342686,  1.3205609,  -1.31793037, -1.57079633,  -1.445])
target1joint = np.array([2.96690151, -1.60088217,  1.63154722, -1.60144448, -1.57070149, -1.74708013])
target1joint = np.array([3.8236942291259766, -1.469771222477295, 2.03717548051943, -2.422732015649313, -1.7380803267108362, -2.466470781956808])
# target1joint = np.array([3.8441805839538574, -1.508592204456665, 2.0356009642230433, -2.378955980340475, -1.7434089819537562, -2.4464884440051478])
# target1joint = np.array([3.8441805839538574, -1.508592204456665, 2.0356009642230433, -2.378955980340475, -1.7434089819537562, -2.4464884440051478])
# target1joint = np.array([2.86539662, -1.37342686,  1.5205609,  -1.91793037, -0.87079633,  -0])
ee_pos, _ = urxkdl.forward(qpos=target1joint)
print("forward ee pos:", ee_pos)
ee_pos = np.array([0.30852322, 0.08026293, 0.99192-0.87  ])
target_orientation = np.array([0, 0, 0.0, 1])
qpos = urxkdl.inverse(target1joint, ee_pos, target_orientation)
print("ik qpos:", qpos)
# -----------------------------



class AprilTag:
    def __init__(self) -> None:
        # ----------- ZED initial ------------
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.camera_fps = 30  # Set fps at 30
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            exit(1)

        # recording_param = sl.RecordingParameters("/home/yi/test_recording/test.svo", sl.SVO_COMPRESSION_MODE.H264)
        # err = self.zed.enable_recording(recording_param)
        # if err != sl.ERROR_CODE.SUCCESS:
        #     print(repr(err))
        #     exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.mat = sl.Mat()
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(self.mat, sl.VIEW.LEFT)
            self.cam_with = self.mat.get_width()
            self.cam_height = self.mat.get_height()
        self.FHD_fx = 1070.09
        self.FHD_fy = 1070.18
        self.FHD_cx = 983.14
        self.FHD_cy = 532.874
        self.FHD_k1 = -0.0533319
        self.FHD_k2 = 0.0260878
        self.FHD_k3 = -0.0104227
        self.FHD_p1 = 0.000340606
        self.FHD_p2 = 0.000133078


        print("complete the ZED2i initialization.")
        # ------------------------------------------
        # fx = 1070.09
        # fy = 1070.18
        # cx = 983.14
        # cy = 532.874
        # k1 = -0.0533319
        # k2 = 0.0260878
        # p1 = 0.000340606
        # p2 = 0.000133078
        # k3 = -0.0104227

        # self.cap = cv2.VideoCapture(4) # camera ID
        #
        # self.cam_with = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #
        # self.fx = 267
        # self.fy = 268

    def stereo_calibration_chessboard(self):
        w = 9
        h = 6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*22.3  # 2.23cm
        obj_points = []
        img_points = []
        num_imgs = 30
        while num_imgs > 1:
            _, img = self.cap.read()
            img_r = img[0:376, 0:672]
            gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
            if ret:
                num_imgs -= 1
                # img_points = np.array(corners)
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
                cv2.drawChessboardCorners(img_r, (w, h), corners, ret)
                print("get the corner:", num_imgs)
                # _, rvec, tvec = cv2,
                cv2.imshow("calibration", img_r)
                time.sleep(1.5)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                cv2.putText(img_r, "unable to detect ChessBoard", (20, img_r.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                cv2.imshow("calibration", img_r)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()
        ret, mtx, dist, rcevs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        Camera_intrinsic = {"mtx": mtx, "dist": dist}
        print(len(img_points))
        print("ret:", ret)
        print("mtx:\n", mtx) # 内参数矩阵
        print("dist:\n", dist)  # 畸变系数
        # print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
        # print("tvecs:\n", tvecs ) # 平移向量  # 外参数



    def tag_dis(self):
        # ----------------- For ZED ------------
        K = np.array([[self.FHD_fx, 0., self.FHD_cx],
                      [0., self.FHD_fy, self.FHD_cy],
                      [0., 0., 1.]])
        K1 = np.array([self.FHD_fx, self.FHD_fy, self.FHD_cx, self.FHD_cy])
        id_root = 10
        id_object = 6
        tag_len = 3.59  # centmeters (smaller size)
        tag_outer_side = 0.91 # distance between the side of tag to outer side
        objoffset = 3
        obj_offset_y = 3.97
        obj_offset_x = 3.2
        obj_offset_z = 2.5
        root_z_offset = 1.25
        root_base_x = 14.5
        root_base_y = 12.83
        rootTrootside = np.identity(4)
        rootsideTcam = np.identity(4)
        camTobjside = np.identity(4)
        objsideTobj = np.identity(4)
        # base_dia = 12.8 + 2 * 0.45
        # rootTrootside[0, 3] = (base_dia / 2 + tag_len / 2 + tag_outer_side)
        rootTrootside[0, 3] = ((root_base_x / 2) - (tag_len / 2 + tag_outer_side))
        rootTrootside[1, 3] = (tag_len / 2 + tag_outer_side + root_base_y / 2)
        rootTrootside[2, 3] = root_z_offset
        objsideTobj[0, 3] = -obj_offset_x
        # objsideTobj[0, 3] = -(tag_len / 2 + tag_outer_side + objoffset)
        objsideTobj[1, 3] = -(tag_len / 2 + tag_outer_side + obj_offset_y)
        objsideTobj[2, 3] = -obj_offset_z
        # --------------------------------------
        cc = cv2.VideoWriter_fourcc(*'XVID')
        file = cv2.VideoWriter('/home/yi/test_output.avi', cc, 20.0, (1920,1080),True)
        j=0
        while True:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                self.zed.retrieve_image(self.mat, sl.VIEW.LEFT)
                img = self.mat.get_data()

                cv2.waitKey(1)
            else:
                cv2.waitKey(1)
            # grabbed, img = self.cap.read()
            # img = img[0:376, 0:672]

            # print(img.shape) #(376, 1344, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
            tags = at_detactor.detect(gray)
            # print(tags)

            for tag in tags:
                H = tag.homography
                # print(H)
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
                r = R.from_matrix(Rs[3].T)
                # print(Ts[0])
                eulerangle = r.as_euler("xyz").T*180/math.pi
                # print(eulerangle)
                # print(Ts)
                # print("num: {}".format(num), "Rs:{}".format(Rs), "Ts:{}".format(Ts), "Ns:{}".format(Ns))
                for i in range(4):
                    cv2.circle(img, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)
                    # print("corner:{}".format(i), tag.corners[i])
                # dis02 = np.linalg.norm(tag.corners[0] - tag.corners[2])
                # real_z = (zed_focal * tag_len) / dis02
                # print(dis03)
                # cv2.circle(img, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)
                # if tag.tag_id == id_root:
                #     root_real_z = np.copy(real_z)
                #     root_center_x = np.copy(np.linalg.norm(tag.center[0] - zed_center_x))
                #     root_center_y = np.copy(np.linalg.norm(tag.center[1] - zed_center_y))
                #     # print("center", tag.center)
                #     root_real_x = (root_center_x * real_z) / zed_focal
                #     root_real_y = (root_center_y * real_z) / zed_focal
                # elif tag.tag_id == id_object:
                #     obj_real_z = np.copy(real_z)
                #     obj_center_x = np.copy(np.linalg.norm(tag.center[0] - zed_center_x))
                #     obj_center_y = np.copy(np.linalg.norm(tag.center[1] - zed_center_y))
                #     obj_real_x = (obj_center_x * real_z) / zed_focal
                #     obj_real_y = (obj_center_y * real_z) / zed_focal
                # print(tag.tag_id)
                # dis_root_obj = np.linalg.norm(np.array([obj_real_x, obj_real_y, obj_real_z]) - np.array([root_real_x, root_real_y, root_real_z]))
                # print(dis_root_obj)

                # dis03 = np.linalg.norm(tag.center[0] - tag.center[0])
                # ---------------------------------------------------
                M, e1, e2 = at_detactor.detection_pose(tag, K1)
                # print(M)
                P = M[:3, :4]
                _t = M[:3, 3]
                t = tag_len * _t
                P = np.matmul(K, P)
                # print(P)
                z = np.matmul(P, np.array([[0], [0], [-1], [1]]))
                # print(z)
                z = z / z[2]
                x = np.matmul(P, np.array([[1], [0], [0], [1]]))
                x = x / x[2]
                y = np.matmul(P, np.array([[0], [-1], [0], [1]]))
                y = y / y[2]
                # y = x[:2].T
                # print(y)
                # print("center", tag.center)
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(x[:2].T, axis=0).astype(int)), (0, 0, 255), 2)
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(y[:2].T, axis=0).astype(int)), (0, 255, 0), 2)
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(z[:2].T, axis=0).astype(int)), (255, 0, 0), 2)
                # -----------------------------------------------------
                # angle_z = eulerangle[2] * math.pi / 180
                # angle_y = eulerangle[1] * math.pi / 180
                # angle_x = eulerangle[0] * math.pi / 180
                # deltax = -math.sin(angle_y) * ARROW_LENGTH
                # deltay = ARROW_LENGTH * math.sin(angle_x)
                # center_z = tag.center + np.array([deltax, deltay])
                M[:3, 3] = t
                if tag.tag_id == id_root:
                    root_pos = np.copy(np.squeeze(t.T))
                    rootsideTcam = np.linalg.inv(M)
                    # print(M)
                elif tag.tag_id == id_object:
                    obj_pos = np.copy(np.squeeze(t.T))
                    camTobjside = np.copy(M)
                # dis_root_obj = np.linalg.norm(root_pos - obj_pos)
                rootsideTobjside = np.matmul(rootsideTcam, camTobjside)
                rootTobjside = np.matmul(rootTrootside, rootsideTobjside)
                rootTobj = np.matmul(rootTobjside, objsideTobj)
                x = rootTobj[0, 3] / 100
                y = -rootTobj[1, 3] / 100
                z = -rootTobj[2, 3] / 100
                print(x, y, z+0.87)
                # print("dis:", dis_root_obj)
                # print(root_pos - obj_pos)


                # # print(newcenter, tag.center)
                # cv2.arrowedLine(img, tuple(tag.center.astype(int)), tuple(center_z.astype(int)), (0, 0, 255), 5)
                # -----------------------------------------------------
            cv2.imshow("camera-image", img)
            file.write(img)
            if cv2.waitKey(1) & 0xFF == ord("j"):
                j += 1
                n = str(j)
                filename = str("./image"+n+".jpg")
                cv2.imwrite(filename, img)

            if cv2.waitKey(1) & 0xFF == 27:
                   break
        file.release()
        cv2.destroyAllWindows()
        self.cap.release()
    def tag_robot_calibration(self):
        # kinematic with urdf
        import gym_envs.envs.ur3_kdl as urkdl
        import gym_envs.envs.robots.robotiq_gripper as robotiq_gripper
        # urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur5e_gripper/ur5e_gripper_real.urdf"
        # urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot.urdf"
        urxkdl = urkdl.URx_kdl(urDH_file)
        # qpos = urxkdl.inverse(current_joint, target_position, target_orientation)
        # ee_pos = urxkdl.forward(qpos=qpos)
        ## real robot connection ------------
        rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")
        rtde_c = rtde_control.RTDEControlInterface("10.42.0.162")
        j_init = np.array([1.31423293, -1.55386663, 1.32749743, -1.34477619, -1.57079633, -0.10715635])
        # print("Creating gripper...")
        # gripper = robotiq_gripper.RobotiqGripper()
        # print("Connecting to gripper...")
        # gripper.connect("10.42.0.162", 63352)
        # print("Activating gripper...")
        # gripper.activate()
        actual_q = rtde_r.getActualQ()
        print("real robot qpos:", actual_q)
        vel = 0.3
        acc = 0.3
        #TODO: calculate distance between the tag biside robot and the hole tag to calibrate the robot with tag
        # real
        # real robot ee pos: [-0.11779326862304741, -0.30405910330223257, 0.12820012463452943]
        # forward ee pos: [0.13310136 0.29000844 0.12850213]
        # [-1.67679806 -1.53791097  1.284411   -1.31729635 -1.57079633 -3.24759439]

        ## init arm ------------
        # target1joint = np.array([-2.9200850168811243, -1.4415864211371918, -2.041741371154785, 5.058547007828512, 1.6527293920516968, 0.25450488924980164])
        # ee_pos = urxkdl.forward(qpos=target1joint)
        # print("forward ee pos:", ee_pos)
        rtde_c.moveJ(target1joint, vel, acc)
        # gripper.move_and_wait_for_pos(203, 255, 255)
        ## get apriltag position ----------
        K = np.array([[self.FHD_fx, 0., self.FHD_cx],
                      [0., self.FHD_fy, self.FHD_cy],
                      [0., 0., 1.]])
        K1 = np.array([self.FHD_fx, self.FHD_fy, self.FHD_cx, self.FHD_cy])
        id_root = 10
        id_object = 6
        tag_len = 3.59  # centmeters (smaller size)
        tag_outer_side = 0.91  # distance between the side of tag to outer side
        objoffset = 3
        obj_offset_y = 3.97
        obj_offset_x = 3
        obj_offset_z = 2.5
        # root_z_offset = -0.3023
        root_z_offset = 2.5
        root_base_x = 14.5
        root_base_y = 12.83
        rootTrootside = np.identity(4)
        rootsideTcam = np.identity(4)
        camTobjside = np.identity(4)
        objsideTobj = np.identity(4)
        # base_dia_x = 10
        # base_dia_y = 10
        # base_dia_z = 2.1
        # rootTrootside[0, 3] = (base_dia_x + tag_len / 2 + tag_outer_side)
        # rootTrootside[1, 3] = (base_dia_y + tag_len / 2 + tag_outer_side)
        # rootTrootside[2, 3] = base_dia_z
        # # objsideTobj[0, 3] = -(tag_len / 2 + tag_outer_side + objoffset)
        # objsideTobj[1, 3] = -(tag_len / 2 + tag_outer_side + obj_offset_y)
        # objsideTobj[2, 3] = -obj_offset_z
        rootTrootside[0, 3] = ((root_base_x / 2) - (tag_len / 2 + tag_outer_side))
        rootTrootside[1, 3] = (tag_len / 2 + tag_outer_side + root_base_y / 2)
        rootTrootside[2, 3] = root_z_offset
        # objsideTobj[0, 3] = -(tag_len / 2 + tag_outer_side + objoffset)
        objsideTobj[1, 3] = -(tag_len / 2 + tag_outer_side + obj_offset_y)
        objsideTobj[2, 3] = -obj_offset_z
        for _ in range(3):
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                self.zed.retrieve_image(self.mat, sl.VIEW.LEFT)
                img = self.mat.get_data()

                cv2.waitKey(1)
            else:
                cv2.waitKey(1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
            tags = at_detactor.detect(gray)
            # print(tags)
            for tag in tags:
                H = tag.homography
                # print(H)
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
                r = R.from_matrix(Rs[3].T)
                M, e1, e2 = at_detactor.detection_pose(tag, K1)
                # print(M)
                P = M[:3, :4]
                _t = M[:3, 3]
                t = tag_len * _t
                P = np.matmul(K, P)
                # print(P)
                z = np.matmul(P, np.array([[0], [0], [-1], [1]]))
                # print(z)
                z = z / z[2]
                x = np.matmul(P, np.array([[1], [0], [0], [1]]))
                x = x / x[2]
                y = np.matmul(P, np.array([[0], [-1], [0], [1]]))
                y = y / y[2]
                M[:3, 3] = t
                if tag.tag_id == id_root:
                    root_pos = np.copy(np.squeeze(t.T))
                    rootsideTcam = np.linalg.inv(M)
                    # print(M)
                elif tag.tag_id == id_object:
                    obj_pos = np.copy(np.squeeze(t.T))
                    camTobjside = np.copy(M)
                # dis_root_obj = np.linalg.norm(root_pos - obj_pos)
                rootsideTobjside = np.matmul(rootsideTcam, camTobjside)
                rootTobjside = np.matmul(rootTrootside, rootsideTobjside)
                rootTobj = np.matmul(rootTobjside, objsideTobj)
                x = rootTobj[0, 3] / 100
                y = -rootTobj[1, 3] / 100
                z = -rootTobj[2, 3] / 100
        current_joint = rtde_r.getActualQ()
        # current_joint = target1joint
        target_position = np.array([x, y, z])
        # target_position = np.array([0.31, 0.08,  0.35 - 0.16807999999999998])
        print("tag dis:", target_position)
        target_orientation = np.array([0.0, 0.0, 0.0, 1])
        qpos = urxkdl.inverse(current_joint, target_position, target_orientation)
        print("ik qpos:", qpos)
        print("real robot ee pos:", rtde_r.getActualTCPPose()[:3])
        print("real robot ee rot:", rtde_r.getActualTCPPose()[3:])

    def robot_FTcompensate(self):
        import gym_envs.envs.ur3_kdl as urkdl
        from gym_envs.utils import quaternion_to_euler, euler_to_quaternion
        urxkdl = urkdl.URx_kdl(urDH_file)
        rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")
        rtde_c = rtde_control.RTDEControlInterface("10.42.0.162")
        testingjoint = np.array([3.5541231632232666, -1.6719452343382777, 1.9017990271197718, -2.115063806573385, -2.2655747572528284, -1.6491158644305628])
        x_angle = 30
        y_angle = 15
        d2r = 3.14/180
        z_force = math.cos(x_angle*d2r)
        x_force = math.cos((90-x_angle)*d2r)
        target2joint = np.array([3.14, -1.57, 1.57, -1.57-x_angle*d2r, -1.57-y_angle*d2r, -1.57])
        vel = 0.8
        acc = 0.8
        # rtde_c.moveJ(target1joint, vel, acc)
        d2r = 3.14/180
        import time
        while True:
            rtde_c.moveJ(target1joint, vel, acc)
            print("current qpos:", rtde_r.getActualQ())
            print("ee pos1:", rtde_r.getActualTCPPose())
            # print("ee quat1:", euler_to_quaternion(rtde_r.getActualTCPPose()[3], rtde_r.getActualTCPPose()[4], rtde_r.getActualTCPPose()[5]))
            # print(rtde_r.getActualQ())
            ee_pos, ee_quat = urxkdl.forward(qpos=rtde_r.getActualQ())
            ee_rot = quaternion_to_euler(ee_quat[0],
                                                 ee_quat[1],
                                                 ee_quat[2],
                                                 ee_quat[3])
            print("ee pos urdf1:", ee_pos, ee_quat, ee_rot)
            eeFT = rtde_r.getActualTCPForce()
            target_ee_pos1 = np.array([0.308, 0.0801, 0.18])
            # self.y_rot += 0.00005
            target_ee_quat1 = np.array([0.0, 0.0, 0.0, 1])
            inv_Test_qpos = urxkdl.inverse(
                rtde_r.getActualQ(),
                target_ee_pos1,
                target_ee_quat1)
            print("inverse test:", inv_Test_qpos)
            z_zero_force = eeFT[2]
            x_zero_force = eeFT[0]
            y_zero_force = eeFT[1]
            print("ee F/T sensor1:", eeFT)
            # rawFT = rtde_r.getFtRawWrench()
            # print("rawFT:", rawFT)
            # payloadKG = rtde_r.getPayload()
            # print("payloadKG:", payloadKG)
            time.sleep(0.1)


            # rtde_c.moveJ(target2joint, vel, acc)
            # ee_pos, ee_quat = urxkdl.forward(qpos=rtde_r.getActualQ())
            # ee_rot = quaternion_to_euler(ee_quat[0],
            #                              ee_quat[1],
            #                              ee_quat[2],
            #                              ee_quat[3])
            # # print("ee pos2:", rtde_r.getActualTCPPose())
            # # print("ee quat2:", euler_to_quaternion(rtde_r.getActualTCPPose()[3], rtde_r.getActualTCPPose()[4], rtde_r.getActualTCPPose()[5]))
            # # print("ee pos urdf2:", ee_pos, ee_quat, ee_rot)
            # eeFT = rtde_r.getActualTCPForce()
            # force_len = 1.01
            # f_zy = z_zero_force * math.cos(y_angle*d2r)
            # f_xy = z_zero_force * math.sin(x_angle*d2r)
            # f_x = f_zy * math.sin(x_angle*d2r)
            # f_z = f_zy * math.cos(x_angle*d2r)
            # f_y = f_xy * math.sin(y_angle*d2r)
            # f_x2 = f_xy * math.cos(y_angle*d2r)
            # z_calculated_force = (z_force * z_zero_force) * force_len
            # x_calculated_force = (x_force * z_zero_force) * force_len - x_zero_force
            # print("calculated force (x-y-z):", -f_x, f_y, f_z)
            # print("ee F/T sensor2:", eeFT)
            # time.sleep(0.5)

cam = AprilTag()
cam.tag_dis()
# cam.stereo_calibration_chessboard()
# cam.tag_robot_calibration()
# cam.robot_FTcompensate()
# filename = "/home/yi/apriltag_source/tag36h11_6.png"

# img = cv2.imread(filename)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

# tags = at_detactor.detect(gray)

# # print(tags)

# for tag in tags:
#     for i in range(4):
#         cv2.circle(img, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)

#     cv2.circle(img, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)

# cv2.imshow("apriltag_test", img)
# cv2.waitKey()

# camera = cv2.VideoCapture(4)
