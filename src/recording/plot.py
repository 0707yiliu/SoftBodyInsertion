import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
 
'''
算术平均滤波法
'''
def ArithmeticAverage(inputs,per):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	mean = []
	for tmp in inputs:
		mean.append(tmp.mean())
	return mean
 
'''
递推平均滤波法
'''
def SlidingAverage(inputs,per):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	tmpmean = inputs[0].mean()
	mean = []
	for tmp in inputs:
		mean.append((tmpmean+tmp.mean())/2)
		tmpmean = tmp.mean()
	return mean
 
'''
中位值平均滤波法
'''
def MedianAverage(inputs,per):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	mean = []
	for tmp in inputs:
		tmp = np.delete(tmp,np.where(tmp==tmp.max())[0],axis = 0)
		tmp = np.delete(tmp,np.where(tmp==tmp.min())[0],axis = 0)
		mean.append(tmp.mean())
	return mean
 
'''
限幅平均滤波法
Amplitude:	限制最大振幅
'''
def AmplitudeLimitingAverage(inputs,per,Amplitude):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	mean = []
	tmpmean = inputs[0].mean()
	tmpnum = inputs[0][0]						#上一次限幅后结果
	for tmp in inputs:
		for index,newtmp in enumerate(tmp):
			if np.abs(tmpnum-newtmp) > Amplitude:
				tmp[index] = tmpnum
			tmpnum = newtmp
		mean.append((tmpmean+tmp.mean())/2)
		tmpmean = tmp.mean()
	return mean
 
'''
一阶滞后滤波法
a:			滞后程度决定因子，0~1
'''
def FirstOrderLag(inputs,a):
	tmpnum = inputs[0]							#上一次滤波结果
	for index,tmp in enumerate(inputs):
		inputs[index] = (1-a)*tmp + a*tmpnum
		tmpnum = tmp
	return inputs
 
'''
加权递推平均滤波法
'''
def WeightBackstepAverage(inputs,per):
	weight = np.array(range(1,np.shape(inputs)[0]+1))			#权值列表
	weight = weight/weight.sum()
 
	for index,tmp in enumerate(inputs):
		inputs[index] = inputs[index]*weight[index]
	return inputs
 
'''
消抖滤波法
N:			消抖上限
'''
def ShakeOff(inputs,N):
	usenum = inputs[0]								#有效值
	i = 0 											#标记计数器
	for index,tmp in enumerate(inputs):
		if tmp != usenum:					
			i = i + 1
			if i >= N:
				i = 0
				inputs[index] = usenum
	return inputs
 
'''
限幅消抖滤波法
Amplitude:	限制最大振幅
N:			消抖上限
'''
def AmplitudeLimitingShakeOff(inputs,Amplitude,N):
	#print(inputs)
	tmpnum = inputs[0]
	for index,newtmp in enumerate(inputs):
		if np.abs(tmpnum-newtmp) > Amplitude:
			inputs[index] = tmpnum
		tmpnum = newtmp
	#print(inputs)
	usenum = inputs[0]
	i = 0
	for index2,tmp2 in enumerate(inputs):
		if tmp2 != usenum:
			i = i + 1
			if i >= N:
				i = 0
				inputs[index2] = usenum
	#print(inputs)
	return inputs




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


# ------------------------------
# T = np.arange(0, 0.5, 1/4410.0)
# num = signal.chirp(T, f0=10, t1 = 0.5, f1=1000.0)
# print(num)
# pl.subplot(2,1,1)
# pl.plot(num)
# result = ArithmeticAverage(num.copy(),30)
 
# #print(num - result)
# pl.subplot(2,1,2)
# pl.plot(result)
# pl.show()
# -------------------------------

from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


obs_file_name = 'vision-touch4mm0508133959_8.npy'
obs_file_name_real = 'vision-touch4mm0508133959_7.npy'
obs_real = np.load(obs_file_name_real)
obs = np.load(obs_file_name)
obs_touch_shape = obs.shape
obs_real_shape = obs_real.shape
# for i in range(obs_real_shape[1]):
# 	obs[:, i] = FirstOrderLag(obs[:, i], 0.99)
wn = 5*5/250 # 截止频率2Hz,采样频率1000Hz
b,a = signal.butter(4,wn,'low')
obs[:, 8] = signal.filtfilt(b,a,obs[:, 8])
for i in range(obs_real_shape[1]):
	obs[:, i] = signal.filtfilt(b,a,obs[:, i])

print(obs_touch_shape)
fig = plt.figure()
for plt_index in range(obs_touch_shape[1]):
    plt.subplot(4,5,plt_index+1)
    if plt_index == 0:
        plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs[:, plt_index], label="sim")
        # plt.plot(np.linspace(0, obs_real_shape[0]-1, obs_real_shape[0]), obs_real[:, plt_index], label="real")
    else:
        plt.plot(np.linspace(0, obs_touch_shape[0]-1, obs_touch_shape[0]), obs[:, plt_index], label="sim")
        # plt.plot(np.linspace(0, obs_real_shape[0]-1, obs_real_shape[0]), obs_real[:, plt_index], label="real")
    # plt.ylim((-50, 50))
plt.legend()
plt.show()
