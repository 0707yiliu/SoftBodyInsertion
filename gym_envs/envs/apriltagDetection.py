# MODEL: AprilTag markers detection and camera calibration (ZED Python API with cv2 lib)
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import pyzed.sl as sl
import cv2
import apriltag
import numpy as np
import matplotlib.pyplot as plt
import sys, os, math, time
from scipy.spatial.transform import Rotation as R

class AprilTag:
    def __init__(self) -> None:

        # ----------- ZED initial ------------
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.camera_fps = 30  # Set fps at 30
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)
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
        # ------------------------------------------
        self.K = np.array([[self.FHD_fx, 0., self.FHD_cx],
                      [0., self.FHD_fy, self.FHD_cy],
                      [0., 0., 1.]])
        self.K1 = np.array([self.FHD_fx, self.FHD_fy, self.FHD_cx, self.FHD_cy])
        # self.cap = cv2.VideoCapture(4) # camera ID
        #
        # self.cam_with = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #
        # self.fx = 267
        # self.fy = 268
        # self.K = np.array([[263.04691686,   0.,         331.23900253],
        #               [  0.,         264.03675933, 197.48212171],
        #               [  0.      ,     0.  ,         1.        ]])
        # self.K1 = np.array([263.04691686,   264.03675933,         331.23900253, 197.48212171])
        self.base_dia = 12.8 + 2*0.45
        self.id_root = 6
        self.id_object = 8
        self.tag_len = 6.955
        self.tag_side = 1.84
        self.objoffset = 3
        root_z_offset = 0.3023

        self.rootTobj = np.identity(4)
        self.rootTrootside = np.identity(4)
        self.rootsideTcam = np.identity(4)
        self.camTobjside = np.identity(4)
        self.objsideTobj = np.identity(4)
        self.rootTrootside[0, 3] = self.base_dia/2 + self.tag_len/2 + self.tag_side
        self.rootTrootside[2, 3] = -root_z_offset
        self.objsideTobj[0, 3] = -(self.tag_len/2 + self.tag_side + self.objoffset)


        self.x = 0
        self.y = 0
        self.z = 0

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

    def test(self):
        # pass
        i = 0
        n = None
        print(self.cam_with)
        print(self.cam_height)
        K = np.array([[263.04691686,   0.,         331.23900253],
                      [  0.,         264.03675933, 197.48212171],
                      [  0.      ,     0.  ,         1.        ]]) 
        K1 = np.array([263.04691686,   264.03675933,         331.23900253, 197.48212171])
        ARROW_LENGTH = 80
        id_root = 6
        id_object = 7
        tag_len = 6.955
        tag_side = 1.84
        objoffset = 1
        zed_focal = K[0,0]
        zed_center_x = K[0,2]
        zed_center_y = K[1,2]
        root_real_x = 0
        root_real_y = 0
        root_real_z = 0
        obj_real_x = 0
        obj_real_y = 0
        obj_real_z = 0
        root_pos = obj_pos = 0
        rootTrootside = rootsideTcam = camTobjside = objsideTobj = np.identity(4)
        base_dia = 12.8
        rootTrootside[0, 3] = base_dia/2 + tag_len/2 + tag_side
        objsideTobj[0, 3] = tag_len/2 + tag_side + objoffset
        while True:
            grabbed, img = self.cap.read()
            img = img[0:376, 0:672]
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
                y = np.matmul(P, np.array([[0], [1], [0], [1]]))
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
                
                if tag.tag_id == id_root:
                    root_pos = np.copy(np.squeeze(t.T))
                    M[:3, 3] = t
                    rootsideTcam = np.linalg.inv(M)
                    # print(M)
                elif tag.tag_id == id_object:
                    obj_pos = np.copy(np.squeeze(t.T))
                    M[:3, 3] = t
                    camTobjside = np.copy(M)
                dis_root_obj = np.linalg.norm(root_pos - obj_pos)
                rootsideTobjside = np.matmul(rootsideTcam, camTobjside)
                rootTobjside = np.matmul(rootTrootside,rootsideTobjside)
                rootTobj = np.matmul(rootTobjside, objsideTobj)
                # print(rootTobj)
                # print("dis:", dis_root_obj)
                # print(root_pos - obj_pos)


                # # print(newcenter, tag.center)
                # cv2.arrowedLine(img, tuple(tag.center.astype(int)), tuple(center_z.astype(int)), (0, 0, 255), 5)
                # -----------------------------------------------------
            cv2.imshow("camera-image", img)
            if cv2.waitKey(1) & 0xFF == ord("j"):
                i += 1
                n = str(i)
                filename = str("./image"+n+".jpg")
                cv2.imwrite(filename, img)

            if cv2.waitKey(1) & 0xFF == 27:
                   break
        cv2.destroyAllWindows()
        self.cap.release()
    
    def calculate_UR3Root2Target(self) -> np.ndarray:
        # cv2 from webcam -----------
        # grabbed, img = self.cap.read()
        # img = img[0:376, 0:672]
        # ----------------------------
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(self.mat, sl.VIEW.LEFT)
            img = self.mat.get_data()
            cv2.waitKey(1)
        else:
            print("can not get the image from ZED.")
            cv2.waitKey(1)
        # print(img.shape) #(376, 1344, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        tags = at_detactor.detect(gray)
        # print(tags)
        for tag in tags:
            # for i in range(4):
            #     cv2.circle(img, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)
            # ---------------------------------------------------
            M, e1, e2 = at_detactor.detection_pose(tag, self.K1)
            # print(M)
            P = M[:3, :4]
            _t = M[:3, 3]
            t = self.tag_len * _t
            P = np.matmul(self.K, P)
            # print(P)
            self.z = np.matmul(P, np.array([[0], [0], [-1], [1]]))
            # print(z)
            self.z = self.z / self.z[2]
            self.x = np.matmul(P, np.array([[1], [0], [0], [1]]))
            self.x = self.x / self.x[2]
            self.y = np.matmul(P, np.array([[0], [-1], [0], [1]]))
            self.y = self.y / self.y[2]
            # y = x[:2].T
            # print(y)
            # print("center", tag.center)
            # -------------------------------
            # cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.x[:2].T, axis=0).astype(int)), (0, 0, 255), 2)
            # cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.y[:2].T, axis=0).astype(int)), (0, 255, 0), 2)
            # cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.z[:2].T, axis=0).astype(int)), (255, 0, 0), 2)
            # -----------------------------------------------------
            M[:3, 3] = t
            if tag.tag_id == self.id_root:
                self.rootsideTcam = np.linalg.inv(M)
            elif tag.tag_id == self.id_object:
                self.camTobjside = np.copy(M)
            rootsideTobjside = np.matmul(self.rootsideTcam, self.camTobjside)
            rootTobjside = np.matmul(self.rootTrootside, rootsideTobjside)
            self.rootTobj = np.matmul(rootTobjside, self.objsideTobj)
            # now we just output x, y, z (unit: meter)
            self.x = self.rootTobj[0, 3] / 100
            self.y = -self.rootTobj[1, 3] / 100
            self.z = -self.rootTobj[2, 3] /100
            # print(x, y, z)
        # cv2.imshow("camera-image", img)  
        return self.x, self.y, self.z
            # print(rootTobj)
            # print("dis:", dis_root_obj)
            # print(root_pos - obj_pos)


            # # print(newcenter, tag.center)
            # cv2.arrowedLine(img, tuple(tag.center.astype(int)), tuple(center_z.astype(int)), (0, 0, 255), 5)
            # -----------------------------------------------------
    #     cv2.imshow("camera-image", img)
    #     if cv2.waitKey(1) & 0xFF == ord("j"):
    #         i += 1
    #         n = str(i)
    #         filename = str("./image"+n+".jpg")
    #         cv2.imwrite(filename, img)

    #     if cv2.waitKey(1) & 0xFF == 27:
    #             break
    # cv2.destroyAllWindows()
    # self.cap.release()


# cam = AprilTag()
# cam.test()
# cam.stereo_calibration_chessboard()
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
