# import matplotlib.pyplot as plt

# import numpy as np

# obs_file_name = '1208102207.npy'
# obs = np.load(obs_file_name)

# obs_touch = obs[:, 6:18]


# obs_touch_shape = obs_touch.shape

# plt.figure(1)

# for plt_index in range(obs_touch_shape[1]):
#     plt.subplot(2,6, plt_index+1)
#     plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs_touch[:, plt_index], label='4mm')


# obs_file_name = 'vision-touch1cm1208112404.npy'
# obs = np.load(obs_file_name)

# obs_touch = obs[:, 6:18]
# obs_touch_shape = obs_touch.shape
# plt.figure(1)

# for plt_index in range(obs_touch_shape[1]):
#     plt.subplot(2,6, plt_index+1)
#     plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs_touch[:, plt_index], label='1cm')

# obs_file_name = 'vision-touch2cm1208112143.npy'
# obs = np.load(obs_file_name)

# obs_touch = obs[:, 6:18]
# obs_touch_shape = obs_touch.shape
# plt.figure(1)

# for plt_index in range(obs_touch_shape[1]):
#     plt.subplot(2,6, plt_index+1)
#     plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs_touch[:, plt_index], label='2cm')

# plt.legend()

# plt.show()


import matplotlib.pyplot as plt

import numpy as np

# obs_file_name = 'touch1cm1212132741.npy'
# obs = np.load(obs_file_name)
# obs_touch = obs[:, 3:]

# obs_force_shape = obs_touch.shape
# ft_title =( 'FX', 'FY', 'FZ', 'TX', 'TY', 'TZ')
# plt.figure(1)

# for plt_index in range(obs_force_shape[1]):
#     plt.subplot(2,3, plt_index+1)

#     plt.plot(np.linspace(0, obs_force_shape[0]-1, obs_force_shape[0]), obs_touch[:, plt_index], label='4mm')
#     plt.title(ft_title[plt_index])
#     if plt_index == 0 or plt_index == 3:
#         plt.ylabel('force') 

# obs_file_name = 'touch2cm1212132519.npy'
# obs = np.load(obs_file_name)

# obs_touch = obs[:, 3:]
# obs_touch_shape = obs_touch.shape
# plt.figure(1)

# for plt_index in range(obs_touch_shape[1]):
#     plt.subplot(2,3, plt_index+1)
#     plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs_touch[:, plt_index], label='1cm')

# obs_file_name = 'touch4mm1212132256.npy'
# obs = np.load(obs_file_name)

# obs_touch = obs[:, 3:]
# obs_touch_shape = obs_touch.shape
# plt.figure(1)

# for plt_index in range(obs_touch_shape[1]):
#     plt.subplot(2,3, plt_index+1)
#     plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs_touch[:, plt_index], label='2cm')

# plt.legend()
# ------------
# obs_file_name = 'vision-touch1mm0221142601_real.npy' # fail
# obs_file_name_real = 'vision-touch1mm0221142730_real.npy' # suc

# obs_file_name = 'vision-touch4mm0221163259_real_nodsl.npy' # fail
# obs_file_name_real = 'vision-touch4mm0221163143_real_nodsl.npy' # suc

obs_file_name = 'vision-touch1mm0228113852_real.npy'
obs_file_name_real = 'vision-touch1mm0228113852_real.npy'
obs_real = np.load(obs_file_name_real)
obs = np.load(obs_file_name)
obs_touch_shape = obs.shape
obs_real_shape = obs_real.shape
print(obs_touch_shape)
fig = plt.figure()
for plt_index in range(obs_touch_shape[1]):
    plt.subplot(3,5,plt_index+1)
    if plt_index == 0:
        plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs[:, plt_index], label="sim")
        plt.plot(np.linspace(0, obs_real_shape[0]-1, obs_real_shape[0]), obs_real[:, plt_index], label="real")
    else:
        plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs[:, plt_index], label="sim")
        plt.plot(np.linspace(0, obs_real_shape[0]-1, obs_real_shape[0]), obs_real[:, plt_index], label="real")
    # plt.ylim((-50, 50))
plt.legend()
plt.show()
