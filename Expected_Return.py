import glob
from pylab import *


num_of_repeats = 10

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
rcParams.update(params)


def load(dir):
    f_list = glob.glob('Data/Expected Return/'+ dir +'/*')
    num_lines = sum(1 for line in open(f_list[0]))
    i = 0
    data = np.zeros((len(f_list), num_lines))
    for i in range(len(f_list)):
        data[i, :] = np.loadtxt(f_list[i])
        i += 1
    return data, len(f_list)


def perc(data, f_list_len):
    median = np.zeros(int(len(data[0])/num_of_repeats))
    perc_25 = np.zeros(int(len(data[0])/num_of_repeats))
    perc_75 = np.zeros(int(len(data[0])/num_of_repeats))
    for i in range(len(median)):
        datai = []
        for k in range(f_list_len):
            datai.extend(data[k,:][i*num_of_repeats:num_of_repeats+i*num_of_repeats])
        median[i] = np.median(datai)
        perc_25[i] = np.percentile(datai, 25)
        perc_75[i] = np.percentile(datai, 75)
    
    return median, perc_25, perc_75


data_td3, t_len = load('TD3')
data_sac, s_len = load('SAC')
#data_ppo, d_len = load('PPO')

# generate the x
rows = []
rows.append(data_td3.shape[1])
rows.append(data_sac.shape[1])
#rows.append(data_ppo.shape[1])
x = np.arange(0, max(rows)/num_of_repeats)

# compute the medians and 25/75 percentiles
med_td3, perc_25_td3, perc_75_td3 = perc(data_td3, t_len)
med_sac, perc_25_sac, perc_75_sac = perc(data_sac, s_len)
#med_ppo, perc_25_ppo, perc_75_ppo = perc(data_ppo, d_len)

axes(frameon=0)
grid()

fill_between(x, perc_25_td3, perc_75_td3, alpha=0.25, linewidth=0, color='#B22400')
fill_between(x, perc_25_sac, perc_75_sac, alpha=0.25, linewidth=0, color='#006BB2')
#fill_between(x, perc_25_ppo, perc_75_ppo, alpha=0.25, linewidth=0, color='#006BB2')


plot(x, med_td3, linewidth=2, color='#B22400')
plot(x, med_sac, linewidth=2, linestyle='--', color='#006BB2')
#plot(x, med_ppo, linewidth=2, linestyle='--', color='#006BB2')

xlim(0, 20)
ylim(-30000, -1000)

xticks(np.arange(0, max(rows)/num_of_repeats, 10))

legend = legend(["TD3", "SAC", "PPO"], loc=4)
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('0.9')

show()