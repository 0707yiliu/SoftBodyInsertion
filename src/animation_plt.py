import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import time
# for vision-touch-dsl 12-dimension
fig, ax = plt.subplots(6, 2, figsize=(9, 10.5))
data_file = list()
data_file.append("vision-touch1mm0304132908_real1.npy")
data_file.append("vision-touch1mm0304133236_real2.npy")
data_file.append("vision-touch1mm0304133426_real3.npy")
data_file.append("vision-touch1mm0304135310_real6.npy")

obj_list = ["circle_4mm", "liloutriangle_4mm", "liloutrianglebig_1mm", "triangle1mm", "triangle2_4mm", "triangle2big_1mm", "triangle2big_4mm", "triangle4mm", "trianglelack_1mm"]
model = "vision-touch_"
obj = obj_list[-2] # chose the obj
model_file = model + obj + "/"
root_path = "obs_animation/"
# root_path = ""
# G:\\UGent Lab Transfer\\video\\state_\\
obs_0 = np.load(root_path+model_file+data_file[0])
obs_1 = np.load(root_path+model_file+data_file[1])
obs_2 = np.load(root_path+model_file+data_file[2])
obs_3 = np.load(root_path+model_file+data_file[3])

obs_max_len = max(obs_0.shape[0], obs_1.shape[0], obs_2.shape[0], obs_3.shape[0]) + 10
# print(obs_max_len)

obs_0 = np.pad(obs_0, ((0, obs_max_len - obs_0.shape[0]),(0,0)), 'constant', constant_values=(0, 0))
obs_1 = np.pad(obs_1, ((0, obs_max_len - obs_1.shape[0]),(0,0)), 'constant', constant_values=(0, 0))
obs_2 = np.pad(obs_2, ((0, obs_max_len - obs_2.shape[0]),(0,0)), 'constant', constant_values=(0, 0))
obs_3 = np.pad(obs_3, ((0, obs_max_len - obs_3.shape[0]),(0,0)), 'constant', constant_values=(0, 0))

obs = []
obs.append(obs_0)
obs.append(obs_1)
obs.append(obs_2)
obs.append(obs_3)
obs = np.array(obs)
# print(obs.shape)
ylabels_vision_touch = ["EEF-x", "EEF-y", "F-x", "F-y", "F-z", "tau-x", "tau-y", "tau-z", "EEF-joint", "hole-x", "hole-y", "hole-z"]
color_set = ["green", "red", "blue", "grey"]
x = np.arange(0, 2*np.pi, 0.01)

# the totally number of line
num_line = 12
num_set = 4
line = []
for i in range(num_line*num_set):
    line.append(Line2D([], []))
# for i in range(4):
#     line[i], = ax[0,1].plot(np.linspace(0, obs_max_len - 1, obs_max_len), obs[i, :, 1], color=color_set[i], label='eef-x')
for j in range(num_set):
    # 4 datasets
    for k in range(num_line):
        # 12 types of data in each dataset
        # ax[k].plot(np.linspace(0, obs_max_len - 1, obs_max_len), obs[j, :, k], color=color_set[j], label='eef-x')
        if k <= 5:
            x = k
            y = 0
        else:
            x = k - 6
            y = 1
        # ax[x][y].clear()
        item = k + j * num_line

        line[item], = ax[x,y].plot(np.linspace(0, obs_max_len - 1, obs_max_len), obs[j, :, k], color=color_set[j])
        ax[x,y].set_ylabel(ylabels_vision_touch[k])
        # print((j+1)*(k+1)-1)
        # ydata[j, i, k] = np.copy(obs[j, i, k])
        if k == 5 or k == 11:
            ax[x, y].set_xlabel("Step")
        if k == 2 or k == 3 or k == 4 or k == 5 or k == 6 or k == 7:
            ax[x,y].set_ylim([-2.3, 2.3])
        else:
            ax[x,y].set_ylim([-1.2, 1.2])

        # print(item)

# time.sleep(10)
        # ax[x,y].set_xlim([0, obs_max_len+10])
# line00, = ax[0,1].plot(x, np.sin(x))
# line01, = ax[1][1].plot(x, np.cos(x))
# line02, = ax[2][1].plot(x, np.cos(x))
# line03, = ax[3][1].plot(x, np.cos(x))
# line04, = ax[4][1].plot(x, np.cos(x))
# line05, = ax[0,0].plot(x, np.cos(x))
# line06, = ax[1][0].plot(x, np.cos(x))
# line07, = ax[2][0].plot(x, np.cos(x))
# line08, = ax[3][0].plot(x, np.cos(x))
# line09, = ax[4][0].plot(x, np.cos(x))


# ydata = np.zeros((4, obs_max_len, 12))


def init_ani():
    for j in range(num_set):
        # 4 datasets
        for k in range(num_line):
            item = k + j * num_line
            line[item].set_data([],[])

def animate(i):
    print(i)
    # # ax.clear()
    # plt.cla()
    for j in range(num_set):
        # 4 datasets
        for k in range(num_line):
            # 12 types of data in each dataset
            
            # ax[k].plot(np.linspace(0, obs_max_len - 1, obs_max_len), obs[j, :, k], color=color_set[j], label='eef-x')
            # if k <= 5:
            #     x = k
            #     y = 0
            # else:
            #     x = k - 6
            #     y = 1
            item = k + j * num_line
            line[item].set_data(np.linspace(0, i, i), obs[j, :i, k])
    #         # ax[x][y].clear()
    #         ln, = ax[x,y].plot(np.linspace(0, i, i), obs[j, :i, k], linewidth=1, color=color_set[j], label='eef-x')
    #         # ydata[j, i, k] = np.copy(obs[j, i, k])
    #         # # print(type(ydata[j, :i, k].tolist()))
    #         # ln.set_ydata(ydata[j, :, k])
    #         # print(np.linspace(0, obs_max_len - 1, obs_max_len)[i])
    #         if k == 2 or 3 or 4 or 5 or 6 or 7:
    #             ax[x,y].set_ylim([-2.3, 2.3])
    #         else:
    #             ax[x,y].set_ylim([-1.2, 1.2])
    return line
    

    # line00.set_ydata(np.sin(x + i/50.0))  # update the data
    # line01.set_ydata(np.cos(x + i/100.0))
    # line02.set_ydata(np.cos(x + i/100.0))
    # line03.set_ydata(np.cos(x + i/100.0))
    # line04.set_ydata(np.cos(x + i/100.0))
    # line05.set_ydata(np.cos(x + i/100.0))
    # line06.set_ydata(np.cos(x + i/100.0))
    # line07.set_ydata(np.cos(x + i/100.0))
    # line08.set_ydata(np.cos(x + i/100.0))
    # line09.set_ydata(np.cos(x + i/100.0))

    # return line00,line01


# # Init only required for blitting to give a clean slate.
# def init():
#     line00.set_ydata(np.sin(x))
#     line01.set_ydata(np.sin(x))
#     return line00,line01

# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
ani = animation.FuncAnimation(fig=fig, func=animate, frames=obs_max_len, init_func=init_ani,
                              interval=100, repeat=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
ani.save(obj + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
 
# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(5,5)
# points = [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)]
# def animate(i):
#     # ax.clear()
#     # Get the point from the points list at index i
#     point = points[i]
#     # Plot that point using the x and y coordinates
#     ax.plot(point[0], point[1], color='green', 
#             label='original', marker='o')
#     # Set the x and y axis to display a fixed range
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
# ani = FuncAnimation(fig, animate, frames=len(points),
#                     interval=500, repeat=True)
# plt.show()