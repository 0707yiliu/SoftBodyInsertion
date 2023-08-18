import os
import cv2

# image path
root = "/home/yi/project_ghent/recording/"
imgs_dir = 'visiontouch0807135502_real_nodsl' # the images root directory, you can modify it to change the images source
im_dir = root + imgs_dir + "/jpgsource/"
# output video path
save_video_dir = root + imgs_dir + "/"
if not os.path.exists(save_video_dir):
    os.makedirs(save_video_dir)
# set saved fps
fps = 20
# get frames list
# frames = sorted(os.listdir(im_dir))
frames = os.listdir(im_dir)
frames.sort(key=lambda x:int(x.split('.')[0]))
# w,h of image
img = cv2.imread(os.path.join(im_dir, frames[0]))
img_size = (img.shape[1], img.shape[0])
# get seq name
seq_name = os.path.dirname(save_video_dir).split('/')[-1]
# splice video_dir
video_dir = os.path.join(save_video_dir, seq_name + '.mp4')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames:
    f_path = os.path.join(im_dir, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + " has been written!")

videowriter.release()
