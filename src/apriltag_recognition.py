import cv2
import apriltag
import numpy as np
import matplotlib.pyplot as plt
import sys, os, math, time
from scipy.spatial.transform import Rotation as R

class AprilTag:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(4) # camera ID

        self.cam_with = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.fx = 267
        self.fy = 268

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
        K1 = np.array([263.04691686,  264.03675933, 331.23900253, 197.48212171])
        ARROW_LENGTH = 80
        id_root = 6
        id_object = 7

        tag_len = 6.955 # centmeters
        tag_side = 1.84
        objoffset = 3
        root_z_offset = -0.3023

        zed_focal = K[0,0]
        zed_center_x = K[0,2]
        zed_center_y = K[1,2]
        root_real_x = 0
        root_real_y = 0
        root_real_z = 0
        obj_real_x = 0
        obj_real_y = 0
        obj_real_z = 0
        root_pos = 0 
        obj_pos = 0
        rootTrootside = np.identity(4)
        rootsideTcam = np.identity(4)
        camTobjside = np.identity(4)
        objsideTobj = np.identity(4)
        base_dia = 12.8 + 2 * 0.45
        rootTrootside[0, 3] = (base_dia/2 + tag_len/2 + tag_side)
        rootTrootside[2, 3] = root_z_offset
        objsideTobj[0, 3] = -(tag_len/2 + tag_side + objoffset)
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
                rootTobjside = np.matmul(rootTrootside,rootsideTobjside)
                rootTobj = np.matmul(rootTobjside, objsideTobj)
                x = -rootTobj[0, 3] / 100
                y = rootTobj[1, 3] / 100
                z = -rootTobj[2, 3] / 100
                print(x, y, z)
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
cam = AprilTag()
cam.test()
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
