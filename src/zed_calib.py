import pyzed.sl as sl
import cv2
import numpy as np
def main():
    # Create a Camera object
    zed = sl.Camera()
    # zed.get_camera_information()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Capture 50 frames and stop
    i = 0
    runtime_parameters = sl.RuntimeParameters()
    mat = sl.Mat()
    # image_zed = sl.Mat(1920, 1080, sl.MAT_TYPE.U8_C4)
    # image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    # # Retrieve data in a numpy array with get_data()
    # image_ocv = image_zed.get_data()
    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            # cv2.imshow("ZED", mat.get_data()) # Get the timestamp at the time the image was captured
            # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
            #       timestamp.get_milliseconds()))
            # i = i + 1

            # Display the left image from the numpy array

            # view = np.concatenate(cv2.resize(image_ocv, (1920, 1080)))

            cv2.imshow("Image", mat.get_data())
            cv2.waitKey(1)

    # Close the camera
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()


# import pyzed.sl as sl
# import cv2
# import numpy as np
# def main():
#     # Create a Camera object
#     print("Running...")
#     init = sl.InitParameters()
#     init.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
#     init.camera_fps = 30
#     cam = sl.Camera()
#     if not cam.is_opened():
#         print("Opening ZED Camera...")
#     status = cam.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         print(repr(status))
#         exit()
#
#     runtime = sl.RuntimeParameters()
#     mat = sl.Mat()
#
#     # print_camera_information(cam)
#     # print_help()
#
#     key = ''
#     while key != 113:  # for 'q' key
#         err = cam.grab(runtime)
#         if err == sl.ERROR_CODE.SUCCESS:
#             cam.retrieve_image(mat, sl.VIEW.LEFT)
#             cv2.imshow("ZED", mat.get_data())
#             key = cv2.waitKey(5)
#             # settings(key, cam, runtime, mat)
#         else:
#             key = cv2.waitKey(5)
#     cv2.destroyAllWindows()
#
#     cam.close()
#     print("\nFINISH")
#
# if __name__ == "__main__":
#     main()