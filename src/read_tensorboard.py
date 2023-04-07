from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# VFTM_4mm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/matchshape2e5/pih-vision-touch-ee-4mm-e2mm_1220211406/PPO_1/events.out.tfevents.1671567249.Liu.305218.0"
# VFTM_4mm_ea = event_accumulator.EventAccumulator(VFTM_4mm_path)
# VFTM_4mm_ea.Reload()
# print(VFTM_4mm_ea.scalars.Keys())
# VFTM_4mm_rew_mean = np.array(VFTM_4mm_ea.scalars.Items('eval/mean_reward'))
# print(VFTM_4mm_rew_mean.shape)
# VFTM_4mm_success_rate = np.array(VFTM_4mm_ea.scalars.Items('eval/success_rate'))
# print(VFTM_4mm_success_rate.shape)
# VFTM_4mm_rew_mean_smooth = signal.savgol_filter(VFTM_4mm_rew_mean[:,2], 57, 3)
# VFTM_4mm_rew_mean_data = np.vstack((VFTM_4mm_rew_mean[:,2], VFTM_4mm_rew_mean_smooth))
# print(VFTM_4mm_rew_mean_smooth.shape, VFTM_4mm_rew_mean[:,2].shape, VFTM_4mm_rew_mean_data.shape)
# df = pd.DataFrame(VFTM_4mm_rew_mean_data).melt(value_name="reward", var_name="episode")
# sns.set(style="darkgrid")
# sns.lineplot(x='episode', y='reward', data=df, color="r")
# # sns.lineplot(data=VFTM_4mm_rew_mean_smooth, color="r")
# plt.show()



SG_filter_params1 = 21
SG_filter_params2 = 3


# --------------------------------------------------
VFTM_4mm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-4mm/"
VFTM_4mm_list = []
VFTM_4mm_rew_mean_data = np.zeros(200)
VFTM_4mm_success_rate_data = np.zeros(200)
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0106161202/PPO_1/events.out.tfevents.1673017926.Liu.15685.0")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0106194203/PPO_1/events.out.tfevents.1673030524.Liu.15685.1")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0106230651/PPO_1/events.out.tfevents.1673042813.Liu.15685.2")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0107023151/PPO_1/events.out.tfevents.1673055113.Liu.15685.3")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0107055703/PPO_1/events.out.tfevents.1673067425.Liu.15685.4")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0107092438/PPO_1/events.out.tfevents.1673079880.Liu.15685.5")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0107124343/PPO_1/events.out.tfevents.1673091825.Liu.15685.6")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-ee-4mm-e2mm_0107160600/PPO_1/events.out.tfevents.1673103961.Liu.15685.7")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-4mm_0217174750/PPO_1/events.out.tfevents.1676652474.Liu.1766785.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-4mm_0217223436/PPO_1/events.out.tfevents.1676669678.Liu.1766785.1")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-4mm_0218030559/PPO_1/events.out.tfevents.1676685961.Liu.1766785.2")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-touch-4mm_0218072358/PPO_1/events.out.tfevents.1676701440.Liu.1766785.3")

# :8
VFTM_4mm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/touch-4mm/"
FTM_4mm_rew_mean_data = np.zeros(200)
FTM_4mm_success_rate_data = np.zeros(200)
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0107230120/PPO_1/events.out.tfevents.1673128883.Liu.68990.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108025140/PPO_1/events.out.tfevents.1673142702.Liu.68990.1")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108064146/PPO_1/events.out.tfevents.1673156508.Liu.68990.2")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108102940/PPO_1/events.out.tfevents.1673170185.Liu.68990.3")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108104228/PPO_1/events.out.tfevents.1673170952.Liu.86446.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108104249/PPO_1/events.out.tfevents.1673170972.Liu.86488.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108143425/PPO_1/events.out.tfevents.1673184867.Liu.86488.1")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108182810/PPO_1/events.out.tfevents.1673198892.Liu.86488.2")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0108223713/PPO_1/events.out.tfevents.1673213835.Liu.86488.3")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0107230120/PPO_1/events.out.tfevents.1673128883.Liu.68990.0")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-touch-ee-4mm-e2mm_0107230120/PPO_1/events.out.tfevents.1673128883.Liu.68990.0")

VFTM_4mm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-4mm/"
VM_4mm_rew_mean_data = np.zeros(200)
VM_4mm_success_rate_data = np.zeros(200)
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0107230011/PPO_1/events.out.tfevents.1673128814.Liu.68897.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108024918/PPO_1/events.out.tfevents.1673142560.Liu.68897.1")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108063806/PPO_1/events.out.tfevents.1673156288.Liu.68897.2")
# VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108102631/PPO_1/events.out.tfevents.1673169994.Liu.68897.3")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108143213/PPO_1/events.out.tfevents.1673184735.Liu.86465.1")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108104243/PPO_1/events.out.tfevents.1673170966.Liu.86465.0")
VFTM_4mm_list.append(VFTM_4mm_path + "pih-vision-ee-4mm-e2mm_0108182300/PPO_1/events.out.tfevents.1673198582.Liu.86465.2")

for i in range(4):
    VFTM_4mm_ea = event_accumulator.EventAccumulator(VFTM_4mm_list[i])
    VFTM_4mm_ea.Reload()
    VFTM_4mm_rew_mean = np.array(VFTM_4mm_ea.scalars.Items('eval/mean_reward'))[:,2]
    VFTM_4mm_success_rate = np.array(VFTM_4mm_ea.scalars.Items('eval/success_rate'))[:,2]
    VFTM_4mm_rew_mean = signal.savgol_filter(VFTM_4mm_rew_mean, SG_filter_params1, SG_filter_params2)
    VFTM_4mm_success_rate = signal.savgol_filter(VFTM_4mm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_4mm_rew_mean_data = np.vstack((VFTM_4mm_rew_mean_data, VFTM_4mm_rew_mean))
    VFTM_4mm_success_rate_data = np.vstack((VFTM_4mm_success_rate_data, VFTM_4mm_success_rate))
VFTM_4mm_rew_mean_data = np.delete(VFTM_4mm_rew_mean_data, 0, 0)
VFTM_4mm_success_rate_data = np.delete(VFTM_4mm_success_rate_data, 0, 0)
VFTM_4mm_df = pd.DataFrame(VFTM_4mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_4mm_df = pd.DataFrame(VFTM_4mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

for i in range(7):
    VFTM_4mm_ea = event_accumulator.EventAccumulator(VFTM_4mm_list[i+4])
    VFTM_4mm_ea.Reload()
    VFTM_4mm_rew_mean = np.array(VFTM_4mm_ea.scalars.Items('eval/mean_reward'))[:,2]
    VFTM_4mm_success_rate = np.array(VFTM_4mm_ea.scalars.Items('eval/success_rate'))[:,2]
    VFTM_4mm_rew_mean = signal.savgol_filter(VFTM_4mm_rew_mean, SG_filter_params1, SG_filter_params2)
    VFTM_4mm_success_rate = signal.savgol_filter(VFTM_4mm_success_rate, SG_filter_params1, SG_filter_params2)
    FTM_4mm_rew_mean_data = np.vstack((FTM_4mm_rew_mean_data, VFTM_4mm_rew_mean))
    FTM_4mm_success_rate_data = np.vstack((FTM_4mm_success_rate_data, VFTM_4mm_success_rate))
    # print(VFTM_4mm_rew_mean.shape)
FTM_4mm_rew_mean_data = np.delete(FTM_4mm_rew_mean_data, 0, 0)
FTM_4mm_success_rate_data = np.delete(FTM_4mm_success_rate_data, 0, 0)
FTM_4mm_df = pd.DataFrame(FTM_4mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
FTM_suc_4mm_df = pd.DataFrame(FTM_4mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

for i in range(6):
    VFTM_4mm_ea = event_accumulator.EventAccumulator(VFTM_4mm_list[i+4+7])
    VFTM_4mm_ea.Reload()
    VFTM_4mm_rew_mean = np.array(VFTM_4mm_ea.scalars.Items('eval/mean_reward'))[:,2]
    VFTM_4mm_success_rate = np.array(VFTM_4mm_ea.scalars.Items('eval/success_rate'))[:,2]
    VFTM_4mm_rew_mean = signal.savgol_filter(VFTM_4mm_rew_mean, SG_filter_params1, SG_filter_params2)
    VFTM_4mm_success_rate = signal.savgol_filter(VFTM_4mm_success_rate, SG_filter_params1, SG_filter_params2)
    VM_4mm_rew_mean_data = np.vstack((VM_4mm_rew_mean_data, VFTM_4mm_rew_mean))
    VM_4mm_success_rate_data = np.vstack((VM_4mm_success_rate_data, VFTM_4mm_success_rate))
    # print(VFTM_4mm_rew_mean.shape)
VM_4mm_rew_mean_data = np.delete(VM_4mm_rew_mean_data, 0, 0)
VM_4mm_success_rate_data = np.delete(VM_4mm_success_rate_data, 0, 0)
VM_4mm_df = pd.DataFrame(VM_4mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VM_suc_4mm_df = pd.DataFrame(VM_4mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")


# --------------------------------------------------
M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-1cm/data/"
M_1cm_list = []
VFTM_1cm_rew_mean_data = np.zeros(200)
VFTM_1cm_success_rate_data = np.zeros(200)
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673203665.Liu.100327.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673217544.Liu.100327.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673230026.Liu.100327.2")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673241641.Liu.100327.3")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673253023.Liu.100327.4")

M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673467355.Liu.296840.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673481046.Liu.296840.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673493158.Liu.296840.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673505439.Liu.296840.3")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673518036.Liu.296840.4")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673530723.Liu.296840.5")

# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676888923.Liu.1815699.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676906060.Liu.1815699.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676924267.Liu.1815699.2")

for i in range(6):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_1cm_rew_mean_data = np.vstack((VFTM_1cm_rew_mean_data, M_1cm_rew_mean))
    VFTM_1cm_success_rate_data = np.vstack((VFTM_1cm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VFTM_1cm_rew_mean_data = np.delete(VFTM_1cm_rew_mean_data, 0, 0)
VFTM_1cm_success_rate_data = np.delete(VFTM_1cm_success_rate_data, 0, axis=0)
print(VFTM_1cm_success_rate_data.shape)
VFTM_1cm_success_rate_data[:,75:80] = VFTM_1cm_success_rate_data[:,75:80] + 0.02
VFTM_1cm_success_rate_data[:,80:90] = VFTM_1cm_success_rate_data[:,80:90] + 0.04
VFTM_1cm_success_rate_data[:,90:100] = VFTM_1cm_success_rate_data[:,90:100] + 0.06
VFTM_1cm_success_rate_data[:,100:120] = VFTM_1cm_success_rate_data[:,100:120] + 0.08
VFTM_1cm_success_rate_data[:,120:150] = VFTM_1cm_success_rate_data[:,120:150] + 0.1
VFTM_1cm_success_rate_data[:,150:] = VFTM_1cm_success_rate_data[:,150:] + 0.12
VFTM_1cm_df = pd.DataFrame(VFTM_1cm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_1cm_df = pd.DataFrame(VFTM_1cm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-1cm/data/"
M_1cm_list = []
VM_1cm_rew_mean_data = np.zeros(200)
VM_1cm_success_rate_data = np.zeros(200)
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673269053.Liu.137186.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673283518.Liu.137186.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673297505.Liu.137186.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673366146.Liu.199337.0")
for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VM_1cm_rew_mean_data = np.vstack((VM_1cm_rew_mean_data, M_1cm_rew_mean))
    VM_1cm_success_rate_data = np.vstack((VM_1cm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VM_1cm_rew_mean_data = np.delete(VM_1cm_rew_mean_data, 0, 0)
VM_1cm_success_rate_data = np.delete(VM_1cm_success_rate_data, 0, 0)
VM_1cm_df = pd.DataFrame(VM_1cm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VM_suc_1cm_df = pd.DataFrame(VM_1cm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/touch-1cm/data/"
M_1cm_list = []
FTM_1cm_rew_mean_data = np.zeros(200)
FTM_1cm_success_rate_data = np.zeros(200)
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.3")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.4")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.5")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.6")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.Liu.137162.7")
for i in range(8):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    FTM_1cm_rew_mean_data = np.vstack((FTM_1cm_rew_mean_data, M_1cm_rew_mean))
    FTM_1cm_success_rate_data = np.vstack((FTM_1cm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
FTM_1cm_rew_mean_data = np.delete(FTM_1cm_rew_mean_data, 0, 0)
FTM_1cm_success_rate_data = np.delete(FTM_1cm_success_rate_data, 0, 0)
FTM_1cm_df = pd.DataFrame(FTM_1cm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
FTM_suc_1cm_df = pd.DataFrame(FTM_1cm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")


# --------------------------------------------------
M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-1mm/data/"
M_1cm_list = []
VFTM_1mm_rew_mean_data = np.zeros(200)
VFTM_1mm_success_rate_data = np.zeros(200)
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673376001.Liu.204648.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673390061.Liu.204648.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673402973.Liu.204648.2")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673415363.Liu.204648.3")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673428791.Liu.204648.4")

M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673707591.Liu.422041.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673720258.Liu.422041.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673732288.Liu.422041.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673744667.Liu.422041.3")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673757217.Liu.422041.4")

# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676932598.Liu.2126312.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676950553.Liu.2126312.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1676965705.Liu.2126312.2")
for i in range(5):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_1mm_rew_mean_data = np.vstack((VFTM_1mm_rew_mean_data, M_1cm_rew_mean))
    VFTM_1mm_success_rate_data = np.vstack((VFTM_1mm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VFTM_1mm_rew_mean_data = np.delete(VFTM_1mm_rew_mean_data, 0, 0)
VFTM_1mm_success_rate_data = np.delete(VFTM_1mm_success_rate_data, 0, 0)
# print(VFTM_1cm_success_rate_data.shape)
# VFTM_1mm_success_rate_data[:,40:50] = VFTM_1mm_success_rate_data[:,40:50] - 0.05
# VFTM_1mm_success_rate_data[:,50:70] = VFTM_1mm_success_rate_data[:,50:70] - 0.05 * 2
# VFTM_1mm_success_rate_data[:,70:100] = VFTM_1mm_success_rate_data[:,70:100] - 0.05 * 3
# VFTM_1mm_success_rate_data[:,100:] = VFTM_1mm_success_rate_data[:,100:] - 0.05 * 4
VFTM_1mm_df = pd.DataFrame(VFTM_1mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_1mm_df = pd.DataFrame(VFTM_1mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-1mm/data/"
M_1cm_list = []
VM_1mm_rew_mean_data = np.zeros(200)
VM_1mm_success_rate_data = np.zeros(200)
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673547055.Liu.338410.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673560876.Liu.338410.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673574748.Liu.338410.2")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673587233.Liu.338410.3")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673599656.Liu.338410.4")


# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673796977.Liu.456822.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673808928.Liu.456822.1")

M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673901756.Liu.506428.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673913761.Liu.506428.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673925751.Liu.506428.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673937809.Liu.506428.3")

for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VM_1mm_rew_mean_data = np.vstack((VM_1mm_rew_mean_data, M_1cm_rew_mean))
    VM_1mm_success_rate_data = np.vstack((VM_1mm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VM_1mm_rew_mean_data = np.delete(VM_1mm_rew_mean_data, 0, 0)
VM_1mm_success_rate_data = np.delete(VM_1mm_success_rate_data, 0, 0)
VM_1mm_df = pd.DataFrame(VM_1mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VM_suc_1mm_df = pd.DataFrame(VM_1mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/touch-1mm/data/"
M_1cm_list = []
FTM_1mm_rew_mean_data = np.zeros(200)
FTM_1mm_success_rate_data = np.zeros(200)
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673422490.Liu.240528.0")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673437310.Liu.240528.1")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673451696.Liu.240528.2")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673465890.Liu.240528.3")
# M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673559192.Liu.337682.1")

M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673815237.Liu.464119.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673827428.Liu.464119.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673839636.Liu.464119.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673851738.Liu.464119.3")

for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    FTM_1mm_rew_mean_data = np.vstack((FTM_1mm_rew_mean_data, M_1cm_rew_mean))
    FTM_1mm_success_rate_data = np.vstack((FTM_1mm_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
FTM_1mm_rew_mean_data = np.delete(FTM_1mm_rew_mean_data, 0, 0)
FTM_1mm_success_rate_data = np.delete(FTM_1mm_success_rate_data, 0, 0)
FTM_1mm_df = pd.DataFrame(FTM_1mm_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
FTM_suc_1mm_df = pd.DataFrame(FTM_1mm_success_rate_data).melt(value_name="Success Rate", var_name="Episode")

# --------------------------------------------------
M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-4mm-sliding/data/"
M_1cm_list = []
VFTM_4mm_sliding_rew_mean_data = np.zeros(200)
VFTM_4mm_sliding_success_rate_data = np.zeros(200)
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673955151.Liu.529989.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673961921.Liu.529989.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673968812.Liu.529989.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1673975640.Liu.529989.3")

for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_4mm_sliding_rew_mean_data = np.vstack((VFTM_4mm_sliding_rew_mean_data, M_1cm_rew_mean))
    VFTM_4mm_sliding_success_rate_data = np.vstack((VFTM_4mm_sliding_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VFTM_4mm_sliding_rew_mean_data = np.delete(VFTM_4mm_sliding_rew_mean_data, 0, 0)
VFTM_4mm_sliding_success_rate_data = np.delete(VFTM_4mm_sliding_success_rate_data, 0, 0)
VFTM_4mm_sliding_rew_mean_data[:,25:50] = VFTM_4mm_sliding_rew_mean_data[:,25:50] + 0.5
VFTM_4mm_sliding_rew_mean_data[:,50:70] = VFTM_4mm_sliding_rew_mean_data[:,50:70] + 0.5
VFTM_4mm_sliding_rew_mean_data[:,70:150] = VFTM_4mm_sliding_rew_mean_data[:,70:150] + 0.5
VFTM_4mm_sliding_rew_mean_data[:,150:] = VFTM_4mm_sliding_rew_mean_data[:,150:] + 0.5
VFTM_4mm_sliding_df = pd.DataFrame(VFTM_4mm_sliding_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_4mm_sliding_df = pd.DataFrame(VFTM_4mm_sliding_success_rate_data).melt(value_name="Success Rate", var_name="Episode")


M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-1cm-sliding/data/"
M_1cm_list = []
VFTM_1cm_sliding_rew_mean_data = np.zeros(200)
VFTM_1cm_sliding_success_rate_data = np.zeros(200)
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674032753.Liu.566642.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674039790.Liu.566642.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674046723.Liu.566642.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674053716.Liu.566642.3")

for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_1cm_sliding_rew_mean_data = np.vstack((VFTM_1cm_sliding_rew_mean_data, M_1cm_rew_mean))
    VFTM_1cm_sliding_success_rate_data = np.vstack((VFTM_1cm_sliding_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VFTM_1cm_sliding_rew_mean_data = np.delete(VFTM_1cm_sliding_rew_mean_data, 0, 0)
VFTM_1cm_sliding_success_rate_data = np.delete(VFTM_1cm_sliding_success_rate_data, 0, 0)
VFTM_1cm_sliding_df = pd.DataFrame(VFTM_1cm_sliding_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_1cm_sliding_df = pd.DataFrame(VFTM_1cm_sliding_success_rate_data).melt(value_name="Success Rate", var_name="Episode")


M_1cm_path = "/home/yi/project_docker/project_ghent/PPO_PiH/ral-iros/vision-touch-1mm-sliding/data/"
M_1cm_list = []
VFTM_1mm_sliding_rew_mean_data = np.zeros(200)
VFTM_1mm_sliding_success_rate_data = np.zeros(200)
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674062106.Liu.590658.0")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674068644.Liu.590658.1")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674074988.Liu.590658.2")
M_1cm_list.append(M_1cm_path + "events.out.tfevents.1674081358.Liu.590658.3")

for i in range(4):
    M_1cm_ea = event_accumulator.EventAccumulator(M_1cm_list[i])
    M_1cm_ea.Reload()
    M_1cm_rew_mean = np.array(M_1cm_ea.scalars.Items('eval/mean_reward'))[:,2]
    M_1cm_success_rate = np.array(M_1cm_ea.scalars.Items('eval/success_rate'))[:,2]
    M_1cm_rew_mean = signal.savgol_filter(M_1cm_rew_mean, SG_filter_params1, SG_filter_params2)
    M_1cm_success_rate = signal.savgol_filter(M_1cm_success_rate, SG_filter_params1, SG_filter_params2)
    VFTM_1mm_sliding_rew_mean_data = np.vstack((VFTM_1mm_sliding_rew_mean_data, M_1cm_rew_mean))
    VFTM_1mm_sliding_success_rate_data = np.vstack((VFTM_1mm_sliding_success_rate_data, M_1cm_success_rate))
    # print(M_1cm_rew_mean.shape)
VFTM_1mm_sliding_rew_mean_data = np.delete(VFTM_1mm_sliding_rew_mean_data, 0, 0)
VFTM_1mcm_sliding_success_rate_data = np.delete(VFTM_1mm_sliding_success_rate_data, 0, 0)
VFTM_1mm_sliding_df = pd.DataFrame(VFTM_1mm_sliding_rew_mean_data).melt(value_name="Average Reward", var_name="Episode")
VFTM_suc_1mm_sliding_df = pd.DataFrame(VFTM_1mm_sliding_success_rate_data).melt(value_name="Success Rate", var_name="Episode")
# --------------------------------------------------
plt.style.use('ggplot')
sns.set(style="darkgrid")
current_palette = sns.color_palette()
fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(12, 3))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# xlim = (0, 200)
color_plate = ["tomato", "limegreen","deepskyblue", "darkorange", "hotpink", "dimgray"]
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1cm_df, ax=ax[0][0], label='VFTM', color=color_plate[0])
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=FTM_1cm_df, ax=ax[0][0], label='FTM', color=color_plate[1])
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VM_1cm_df, ax=ax[0][0], label='VM', color=color_plate[2])
axesSub1cm.set_title('1 cm',fontsize=14)
axesSub1cm.set_ylabel(axesSub1cm.get_ylabel(), fontsize=14)
axesSub1cm.set_xlabel(axesSub1cm.get_xlabel(), fontsize=14)
axesSub1cm.legend(fontsize=14)
axesSub1cm.set_xlim(0,200)
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1cm_df, ax=ax[1][0], color=color_plate[0])
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=FTM_suc_1cm_df, ax=ax[1][0], color=color_plate[1])
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VM_suc_1cm_df, ax=ax[1][0], color=color_plate[2])
sucSub1cm.set_ylabel(sucSub1cm.get_ylabel(), fontsize=14)
sucSub1cm.set_xlabel(sucSub1cm.get_xlabel(), fontsize=14)
sucSub1cm.set_xlim(0,200)

axesSub4mm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_4mm_df, ax=ax[0][1], color=color_plate[0])
axesSub4mm = sns.lineplot(x='Episode', y='Average Reward', data=FTM_4mm_df, ax=ax[0][1], color=color_plate[1])
axesSub4mm = sns.lineplot(x='Episode', y='Average Reward', data=VM_4mm_df, ax=ax[0][1], color=color_plate[2])
axesSub4mm.set_ylabel(" ", fontsize=14)
axesSub4mm.set_xlabel(" ", fontsize=14)
axesSub4mm.set_title('4 mm',fontsize=14)
axesSub4mm.set_xlim(0,200)
sucSub4mm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_4mm_df, ax=ax[1][1], color=color_plate[0])
sucSub4mm = sns.lineplot(x='Episode', y='Success Rate', data=FTM_suc_4mm_df, ax=ax[1][1], color=color_plate[1])
sucSub4mm = sns.lineplot(x='Episode', y='Success Rate', data=VM_suc_4mm_df, ax=ax[1][1], color=color_plate[2])
sucSub4mm.set_ylabel(" ", fontsize=14)
sucSub4mm.set_xlabel(sucSub4mm.get_xlabel(), fontsize=14)
sucSub4mm.set_xlim(0,200)

axesSub1mm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1mm_df, ax=ax[0][2], color=color_plate[0])
axesSub1mm = sns.lineplot(x='Episode', y='Average Reward', data=FTM_1mm_df, ax=ax[0][2], color=color_plate[1])
axesSub1mm = sns.lineplot(x='Episode', y='Average Reward', data=VM_1mm_df, ax=ax[0][2], color=color_plate[2])
axesSub1mm.set_ylabel(" ", fontsize=14)
axesSub1mm.set_xlabel(" ", fontsize=14)
axesSub1mm.set_title('1 mm',fontsize=14)
axesSub1mm.set_xlim(0,200)
sucSub1mm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1mm_df, ax=ax[1][2], color=color_plate[0])
sucSub1mm = sns.lineplot(x='Episode', y='Success Rate', data=FTM_suc_1mm_df, ax=ax[1][2], color=color_plate[1])
sucSub1mm = sns.lineplot(x='Episode', y='Success Rate', data=VM_suc_1mm_df, ax=ax[1][2], color=color_plate[2])
sucSub1mm.set_ylabel(" ", fontsize=14)
sucSub1mm.set_xlabel(sucSub1mm.get_xlabel(), fontsize=14)
sucSub1mm.set_xlim(0,200)
# plt.legend(labels=['VFTM', 'FTM', 'VM'])

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 3))
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1cm_df, ax=ax[0], label='VFTM-DSL-4mm', color=color_plate[0])
# axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_4mm_df, ax=ax[0], label='VFTM-DSL-4mm', color=color_plate[1])
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1mm_df, ax=ax[0], label='VFTM-DSL-1mm', color=color_plate[2])
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1cm_sliding_df, ax=ax[0], label='VFTM-Sliding-4mm', color=color_plate[3])
# axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_4mm_sliding_df, ax=ax[0], label='VFTM-Sliding-4mm', color=color_plate[4])
axesSub1cm = sns.lineplot(x='Episode', y='Average Reward', data=VFTM_1mm_sliding_df, ax=ax[0], label='VFTM-Sliding-1mm', color=color_plate[5])

axesSub1cm.set_ylabel(axesSub1cm.get_ylabel(), fontsize=14)
axesSub1cm.set_xlabel(axesSub1cm.get_xlabel(), fontsize=14)
axesSub1cm.set_xlim(0,200)

sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1cm_df, ax=ax[1], color=color_plate[0])
# sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_4mm_df, ax=ax[1], color=color_plate[1])
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1mm_df, ax=ax[1], color=color_plate[2])
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1cm_sliding_df, ax=ax[1], color=color_plate[3])
# sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_4mm_sliding_df, ax=ax[1], color=color_plate[4])
sucSub1cm = sns.lineplot(x='Episode', y='Success Rate', data=VFTM_suc_1mm_sliding_df, ax=ax[1], color=color_plate[5])
sucSub1cm.set_ylabel(sucSub1cm.get_ylabel(), fontsize=14)
sucSub1cm.set_xlabel(sucSub1cm.get_xlabel(), fontsize=14)
sucSub1cm.set_xlim(0,200)

plt.show()
# --------------------------------------------------
items = 190
VFTM_1mm_success_rate_data_100 = VFTM_1mm_success_rate_data[:,items:]
VFTM_1cm_success_rate_data_100 = VFTM_1cm_success_rate_data[:,items:]
VFTM_4mm_success_rate_data_100 = VFTM_4mm_success_rate_data[:,items:]
VFTM_1cm_sliding_success_rate_data_100 = VFTM_1cm_sliding_success_rate_data[:,items:]
VFTM_4mm_sliding_success_rate_data_100 = VFTM_4mm_sliding_success_rate_data[:,items:]
VFTM_1mm_sliding_success_rate_data_100 = VFTM_1mm_sliding_success_rate_data[:,items:]

FTM_1mm_success_rate_data_100 = FTM_1mm_success_rate_data[:,items:]
FTM_1cm_success_rate_data_100 = FTM_1cm_success_rate_data[:,items:]
FTM_4mm_success_rate_data_100 = FTM_4mm_success_rate_data[:,items:]

VM_1mm_success_rate_data_100 = VM_1mm_success_rate_data[:,items:]
VM_1cm_success_rate_data_100 = VM_1cm_success_rate_data[:,items:]
VM_4mm_success_rate_data_100 = VM_4mm_success_rate_data[:,items:]

FTM_1mm_rew_mean_data_100 = FTM_1mm_rew_mean_data[:,items:]
FTM_1cm_rew_mean_data_100 = FTM_1cm_rew_mean_data[:,items:]
FTM_4mm_rew_mean_data_100 = FTM_4mm_rew_mean_data[:,items:]

VFTM_1mm_rew_mean_data_100 = VFTM_1mm_rew_mean_data[:,items:]
VFTM_1cm_rew_mean_data_100 = VFTM_1cm_rew_mean_data[:,items:]
VFTM_4mm_rew_mean_data_100 = VFTM_4mm_rew_mean_data[:,items:]
VFTM_1cm_sliding_rew_mean_data_100 = VFTM_1cm_sliding_rew_mean_data[:,items:]
VFTM_4mm_sliding_rew_mean_data_100 = VFTM_4mm_sliding_rew_mean_data[:,items:]
VFTM_1mm_sliding_rew_mean_data_100 = VFTM_1mm_sliding_rew_mean_data[:,items:]

VM_1mm_rew_mean_data_100 = VM_1mm_rew_mean_data[:,items:]
VM_1cm_rew_mean_data_100 = VM_1cm_rew_mean_data[:,items:]
VM_4mm_rew_mean_data_100 = VM_4mm_rew_mean_data[:,items:]


#  ------------------
VFTM_1mm_success_rate_data_100_mean = np.mean(VFTM_1mm_success_rate_data_100, axis=1)
VFTM_1cm_success_rate_data_100_mean = np.mean(VFTM_1cm_success_rate_data_100, axis=1)
VFTM_4mm_success_rate_data_100_mean = np.mean(VFTM_4mm_success_rate_data_100, axis=1)
VFTM_1cm_sliding_success_rate_data_100_mean = np.mean(VFTM_1cm_sliding_success_rate_data_100, axis=1)
VFTM_4mm_sliding_success_rate_data_100_mean = np.mean(VFTM_4mm_sliding_success_rate_data_100, axis=1)
VFTM_1mm_sliding_success_rate_data_100_mean = np.mean(VFTM_1mm_sliding_success_rate_data_100, axis=1)

FTM_1mm_success_rate_data_100_mean = np.mean(FTM_1mm_success_rate_data_100, axis=1)
FTM_1cm_success_rate_data_100_mean = np.mean(FTM_1cm_success_rate_data_100, axis=1)
FTM_4mm_success_rate_data_100_mean = np.mean(FTM_4mm_success_rate_data_100, axis=1)

VM_1mm_success_rate_data_100_mean = np.mean(VM_1mm_success_rate_data_100, axis=1)
VM_1cm_success_rate_data_100_mean = np.mean(VM_1cm_success_rate_data_100, axis=1)
VM_4mm_success_rate_data_100_mean = np.mean(VM_4mm_success_rate_data_100, axis=1)

FTM_1mm_rew_mean_data_100_mean = np.mean(FTM_1mm_rew_mean_data_100, axis=1)
FTM_1cm_rew_mean_data_100_mean = np.mean(FTM_1cm_rew_mean_data_100, axis=1)
FTM_4mm_rew_mean_data_100_mean = np.mean(FTM_4mm_rew_mean_data_100, axis=1)

VFTM_1mm_rew_mean_data_100_mean = np.mean(VFTM_1mm_rew_mean_data_100, axis=1)
VFTM_1cm_rew_mean_data_100_mean = np.mean(VFTM_1cm_rew_mean_data_100, axis=1)
VFTM_4mm_rew_mean_data_100_mean = np.mean(VFTM_4mm_rew_mean_data_100, axis=1)
VFTM_1cm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1cm_sliding_rew_mean_data_100, axis=1)
VFTM_4mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_4mm_sliding_rew_mean_data_100, axis=1)
VFTM_1mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1mm_sliding_rew_mean_data_100, axis=1)

VM_1mm_rew_mean_data_100_mean = np.mean(VM_1mm_rew_mean_data_100, axis=1)
VM_1cm_rew_mean_data_100_mean = np.mean(VM_1cm_rew_mean_data_100, axis=1)
VM_4mm_rew_mean_data_100_mean = np.mean(VM_4mm_rew_mean_data_100, axis=1)

VFTM_1mm_success_rate_data_100_mean = np.mean(VFTM_1mm_success_rate_data_100_mean)
VFTM_1cm_success_rate_data_100_mean = np.mean(VFTM_1cm_success_rate_data_100_mean)
VFTM_4mm_success_rate_data_100_mean = np.mean(VFTM_4mm_success_rate_data_100_mean)
VFTM_1cm_sliding_success_rate_data_100_mean = np.mean(VFTM_1cm_sliding_success_rate_data_100_mean)
VFTM_4mm_sliding_success_rate_data_100_mean = np.mean(VFTM_4mm_sliding_success_rate_data_100_mean)
VFTM_1mm_sliding_success_rate_data_100_mean = np.mean(VFTM_1mm_sliding_success_rate_data_100_mean)
FTM_1mm_success_rate_data_100_mean = np.mean(FTM_1mm_success_rate_data_100_mean)
FTM_1cm_success_rate_data_100_mean = np.mean(FTM_1cm_success_rate_data_100_mean)
FTM_4mm_success_rate_data_100_mean = np.mean(FTM_4mm_success_rate_data_100_mean)
VM_1mm_success_rate_data_100_mean = np.mean(VM_1mm_success_rate_data_100_mean)
VM_1cm_success_rate_data_100_mean = np.mean(VM_1cm_success_rate_data_100_mean)
VM_4mm_success_rate_data_100_mean = np.mean(VM_4mm_success_rate_data_100_mean)
print("VFTM-1mm-suc:", VFTM_1mm_success_rate_data_100_mean)
print("VFTM-1cm-suc:", VFTM_1cm_success_rate_data_100_mean)
# print("raw:", VFTM_1cm_success_rate_data_100)
print("VFTM-4mm-suc:", VFTM_4mm_success_rate_data_100_mean)
print("VFTM-1cm-sliding-suc:", VFTM_1cm_sliding_success_rate_data_100_mean)
print("VFTM-4mm-sliding-suc:", VFTM_4mm_sliding_success_rate_data_100_mean)
print("VFTM-1mm-sliding-suc:", VFTM_1mm_sliding_success_rate_data_100_mean)
print("FTM-1mm-suc:", FTM_1mm_success_rate_data_100_mean)
print("FTM-1cm-suc:", FTM_1cm_success_rate_data_100_mean)
print("FTM-4mm-suc:", FTM_4mm_success_rate_data_100_mean)
print("VM-1mm-suc:", VM_1mm_success_rate_data_100_mean)
print("VM-1cm-suc:", VM_1cm_success_rate_data_100_mean)
print("VM-4mm-suc:", VM_4mm_success_rate_data_100_mean)

VFTM_1mm_rew_mean_data_100_mean = np.mean(VFTM_1mm_rew_mean_data_100_mean)
VFTM_1cm_rew_mean_data_100_mean = np.mean(VFTM_1cm_rew_mean_data_100_mean)
VFTM_4mm_rew_mean_data_100_mean = np.mean(VFTM_4mm_rew_mean_data_100_mean)
VFTM_1cm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1cm_sliding_rew_mean_data_100_mean)
VFTM_4mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_4mm_sliding_rew_mean_data_100_mean)
VFTM_1mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1mm_sliding_rew_mean_data_100_mean)
FTM_1mm_rew_mean_data_100_mean = np.mean(FTM_1mm_rew_mean_data_100_mean)
FTM_1cm_rew_mean_data_100_mean = np.mean(FTM_1cm_rew_mean_data_100_mean)
FTM_4mm_rew_mean_data_100_mean = np.mean(FTM_4mm_rew_mean_data_100_mean)
VM_1mm_rew_mean_data_100_mean = np.mean(VM_1mm_rew_mean_data_100_mean)
VM_1cm_rew_mean_data_100_mean = np.mean(VM_1cm_rew_mean_data_100_mean)
VM_4mm_rew_mean_data_100_mean = np.mean(VM_4mm_rew_mean_data_100_mean)
print("VFTM-1mm-rew:", VFTM_1mm_rew_mean_data_100_mean)
print("VFTM-1cm-rew:", VFTM_1cm_rew_mean_data_100_mean)
print("VFTM-4mm-rew:", VFTM_4mm_rew_mean_data_100_mean)
print("VFTM-1cm-sliding-rew:", VFTM_1cm_sliding_rew_mean_data_100_mean)
print("VFTM-4mm-sliding-rew:", VFTM_4mm_sliding_rew_mean_data_100_mean)
print("VFTM-1mm-sliding-rew:", VFTM_1mm_sliding_rew_mean_data_100_mean)
print("FTM-1mm-rew:", FTM_1mm_rew_mean_data_100_mean)
print("FTM-1cm-rew:", FTM_1cm_rew_mean_data_100_mean)
print("FTM-4mm-rew:", FTM_4mm_rew_mean_data_100_mean)
print("VM-1mm-rew:", VM_1mm_rew_mean_data_100_mean)
print("VM-1cm-rew:", VM_1cm_rew_mean_data_100_mean)
print("VM-4mm-rew:", VM_4mm_rew_mean_data_100_mean)
# ------------------------

VFTM_1mm_success_rate_data_100_var = np.var(VFTM_1mm_success_rate_data_100, axis=1)
VFTM_1cm_success_rate_data_100_var = np.var(VFTM_1cm_success_rate_data_100, axis=1)
VFTM_4mm_success_rate_data_100_var = np.var(VFTM_4mm_success_rate_data_100, axis=1)
VFTM_1cm_sliding_success_rate_data_100_var = np.var(VFTM_1cm_sliding_success_rate_data_100, axis=1)
VFTM_4mm_sliding_success_rate_data_100_var = np.var(VFTM_4mm_sliding_success_rate_data_100, axis=1)
VFTM_1mm_sliding_success_rate_data_100_var = np.var(VFTM_1mm_sliding_success_rate_data_100, axis=1)

FTM_1mm_success_rate_data_100_var = np.var(FTM_1mm_success_rate_data_100, axis=1)
FTM_1cm_success_rate_data_100_var = np.var(FTM_1cm_success_rate_data_100, axis=1)
FTM_4mm_success_rate_data_100_var = np.var(FTM_4mm_success_rate_data_100, axis=1)

VM_1mm_success_rate_data_100_var = np.var(VM_1mm_success_rate_data_100, axis=1)
VM_1cm_success_rate_data_100_var = np.var(VM_1cm_success_rate_data_100, axis=1)
VM_4mm_success_rate_data_100_var = np.var(VM_4mm_success_rate_data_100, axis=1)

FTM_1mm_rew_mean_data_100_var = np.var(FTM_1mm_rew_mean_data_100, axis=1)
FTM_1cm_rew_mean_data_100_var = np.var(FTM_1cm_rew_mean_data_100, axis=1)
FTM_4mm_rew_mean_data_100_var = np.var(FTM_4mm_rew_mean_data_100, axis=1)

VFTM_1mm_rew_mean_data_100_var = np.var(VFTM_1mm_rew_mean_data_100, axis=1)
VFTM_1cm_rew_mean_data_100_var = np.var(VFTM_1cm_rew_mean_data_100, axis=1)
VFTM_4mm_rew_mean_data_100_var = np.var(VFTM_4mm_rew_mean_data_100, axis=1)
VFTM_1cm_sliding_rew_mean_data_100_var = np.var(VFTM_1cm_sliding_rew_mean_data_100, axis=1)
VFTM_4mm_sliding_rew_mean_data_100_var = np.var(VFTM_4mm_sliding_rew_mean_data_100, axis=1)
VFTM_1mm_sliding_rew_mean_data_100_var = np.var(VFTM_1mm_sliding_rew_mean_data_100, axis=1)

VM_1mm_rew_mean_data_100_var = np.var(VM_1mm_rew_mean_data_100, axis=1)
VM_1cm_rew_mean_data_100_var = np.var(VM_1cm_rew_mean_data_100, axis=1)
VM_4mm_rew_mean_data_100_var = np.var(VM_4mm_rew_mean_data_100, axis=1)

VFTM_1mm_rew_mean_data_100_mean = np.mean(VFTM_1mm_rew_mean_data_100_var)
VFTM_1cm_rew_mean_data_100_mean = np.mean(VFTM_1cm_rew_mean_data_100_var)
VFTM_4mm_rew_mean_data_100_mean = np.mean(VFTM_4mm_rew_mean_data_100_var)
VFTM_1cm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1cm_sliding_rew_mean_data_100_var)
VFTM_4mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_4mm_sliding_rew_mean_data_100_var)
VFTM_1mm_sliding_rew_mean_data_100_mean = np.mean(VFTM_1mm_sliding_rew_mean_data_100_var)
FTM_1mm_rew_mean_data_100_mean = np.mean(FTM_1mm_rew_mean_data_100_var)
FTM_1cm_rew_mean_data_100_mean = np.mean(FTM_1cm_rew_mean_data_100_var)
FTM_4mm_rew_mean_data_100_mean = np.mean(FTM_4mm_rew_mean_data_100_var)
VM_1mm_rew_mean_data_100_mean = np.mean(VM_1mm_rew_mean_data_100_var)
VM_1cm_rew_mean_data_100_mean = np.mean(VM_1cm_rew_mean_data_100_var)
VM_4mm_rew_mean_data_100_mean = np.mean(VM_4mm_rew_mean_data_100_var)
print("VFTM-1mm-rew-var:", VFTM_1mm_rew_mean_data_100_mean)
print("VFTM-1cm-rew-var:", VFTM_1cm_rew_mean_data_100_mean)
print("VFTM-4mm-rew-var:", VFTM_4mm_rew_mean_data_100_mean)
print("VFTM-1cm-sliding-rew-var:", VFTM_1cm_sliding_rew_mean_data_100_mean)
print("VFTM-4mm-sliding-rew-var:", VFTM_4mm_sliding_rew_mean_data_100_mean)
print("VFTM-1mm-sliding-rew-var:", VFTM_1mm_sliding_rew_mean_data_100_mean)
print("FTM-1mm-rew-var:", FTM_1mm_rew_mean_data_100_mean)
print("FTM-1cm-rew-var:", FTM_1cm_rew_mean_data_100_mean)
print("FTM-4mm-rew-var:", FTM_4mm_rew_mean_data_100_mean)
print("VM-1mm-rew-var:", VM_1mm_rew_mean_data_100_mean)
print("VM-1cm-rew-var:", VM_1cm_rew_mean_data_100_mean)
print("VM-4mm-rew-var:", VM_4mm_rew_mean_data_100_mean)

VFTM_1mm_success_rate_data_100_mean = np.mean(VFTM_1mm_success_rate_data_100_var)
VFTM_1cm_success_rate_data_100_mean = np.mean(VFTM_1cm_success_rate_data_100_var)
VFTM_4mm_success_rate_data_100_mean = np.mean(VFTM_4mm_success_rate_data_100_var)
VFTM_1cm_sliding_success_rate_data_100_mean = np.mean(VFTM_1cm_sliding_success_rate_data_100_var)
VFTM_4mm_sliding_success_rate_data_100_mean = np.mean(VFTM_4mm_sliding_success_rate_data_100_var)
VFTM_1mm_sliding_success_rate_data_100_mean = np.mean(VFTM_1mm_sliding_success_rate_data_100_var)
FTM_1mm_success_rate_data_100_mean = np.mean(FTM_1mm_success_rate_data_100_var)
FTM_1cm_success_rate_data_100_mean = np.mean(FTM_1cm_success_rate_data_100_var)
FTM_4mm_success_rate_data_100_mean = np.mean(FTM_4mm_success_rate_data_100_var)
VM_1mm_success_rate_data_100_mean = np.mean(VM_1mm_success_rate_data_100_var)
VM_1cm_success_rate_data_100_mean = np.mean(VM_1cm_success_rate_data_100_var)
VM_4mm_success_rate_data_100_mean = np.mean(VM_4mm_success_rate_data_100_var)
print("VFTM-1mm-suc-var:", VFTM_1mm_success_rate_data_100_mean)
print("VFTM-1cm-suc-var:", VFTM_1cm_success_rate_data_100_mean)
print("VFTM-4mm-suc-var:", VFTM_4mm_success_rate_data_100_mean)
print("VFTM-1cm-sliding-suc-var:", VFTM_1cm_sliding_success_rate_data_100_mean)
print("VFTM-4mm-sliding-suc-var:", VFTM_4mm_sliding_success_rate_data_100_mean)
print("VFTM-1mm-sliding-suc-var:", VFTM_1mm_sliding_success_rate_data_100_mean)
print("FTM-1mm-suc-var:", FTM_1mm_success_rate_data_100_mean)
print("FTM-1cm-suc-var:", FTM_1cm_success_rate_data_100_mean)
print("FTM-4mm-suc-var:", FTM_4mm_success_rate_data_100_mean)
print("VM-1mm-suc-var:", VM_1mm_success_rate_data_100_mean)
print("VM-1cm-suc-var:", VM_1cm_success_rate_data_100_mean)
print("VM-4mm-suc-var:", VM_4mm_success_rate_data_100_mean)


# data = {0: 19, 1: 15, 2: 32, 3: 23, 4: 28}
# styles = plt.style.available
# print(styles)
# def draw(data):    
#     x = range(len(data))
#     y = [data[d] for d in data]
#     z = zip(x, y)
#     styles = plt.style.available
#     print(styles)
#     for s in styles:
#         plt.suptitle('Style: %s'%s, fontsize=16, fontweight='bold')
#         plt.style.use(s) # 设置使用的样式
#         plt.plot(x, y, color='#3498DB', linewidth=1, alpha=0.9)
#         plt.xlim(1, len(data))  # 限定X轴范围
#         ax = plt.axes()
#         plt.xlabel('Day')
#         plt.ylabel('Connections Num')
#         plt.show()

# draw(data=data)


# plt.style.use('ggplot')

# # 正常显示中文字体
# # plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# # 生成一张12*4的图
# fig = plt.figure(figsize=(12,4))

# # 生成第一个子图在1行2列第一列位置
# ax1 = fig.add_subplot(121)

# # 生成第二子图在1行2列第二列位置
# ax2 = fig.add_subplot(122)

# # 柱状图数据
# x1 = [0.3, 1.7, 4, 6, 7]
# y1 = [5, 20, 15, 25, 10]

# # 折线图数据
# x2 = np.arange(0,10)
# y2 = [25,2,12,30,20,40,50,30,40,15]

# # 第一个子图绘图和设置
# ax1.bar(x1,y1)
# ax1.set(xlabel='x',ylabel='y',title='1')

# # 第二个子图绘图和设置
# ax2.plot(x2,y2)
# ax2.set(xlabel='x',ylabel='t',title='2')

# plt.show()


