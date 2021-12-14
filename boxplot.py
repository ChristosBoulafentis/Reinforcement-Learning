import glob
from pylab import *
from palettable.colorbrewer.qualitative import Set2_7


params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [2.5, 4.5]
}
rcParams.update(params)

def load(dir):
   f_list = glob.glob('Data/Execution Time/' + dir)
   num_lines = sum(1 for line in open(f_list[0]))
   i  = 0
   data = np.zeros((len(f_list), num_lines))
   for f in f_list:
       data[i, :] = np.loadtxt(f)
       i += 1
   return data[0,:], num_lines


td, td_lines = load('TD3-Time')
sac, sac_lines = load('SAC-Time')
ppo, ppo_lines = load('PPO-Time')

fig = figure()
ax = fig.add_subplot(111)


bp = ax.boxplot([td, sac, ppo], notch=0, sym='b+', vert=1, whis=1.5,positions=None, widths=0.6)

############   Frame    ################
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
############   Grid     ###################################
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
############   Colors   ##########################
# colors, as before
from palettable.colorbrewer.qualitative import Set2_7
colors = Set2_7.mpl_colors

for i in range(0, len(bp['boxes'])):
   bp['boxes'][i].set_color(colors[i])
   # we have two whiskers!
   bp['whiskers'][i*2].set_color(colors[i])
   bp['whiskers'][i*2 + 1].set_color(colors[i])
   bp['whiskers'][i*2].set_linewidth(2)
   bp['whiskers'][i*2 + 1].set_linewidth(2)
   # fliers
   # (set allows us to set many parameters at once)
   bp['fliers'][i].set(markerfacecolor=colors[i],
                   marker='o', alpha=0.75, markersize=6,
                   markeredgecolor='none')
   bp['medians'][i].set_color('black')
   bp['medians'][i].set_linewidth(3)
   # and 4 caps to remove
   for c in bp['caps']:
       c.set_linewidth(0)
###########    Fill Boxes  ####################################
for i in range(len(bp['boxes'])):
   box = bp['boxes'][i]
   box.set_linewidth(0)
   boxX = []
   boxY = []
   for j in range(5):
       boxX.append(box.get_xdata()[j])
       boxY.append(box.get_ydata()[j])
       boxCoords = list(zip(boxX,boxY))
       boxPolygon = Polygon(boxCoords, facecolor = colors[i], linewidth=0)
       ax.add_patch(boxPolygon)
###################################################
fig.subplots_adjust(left=0.2)
ax.set_xticklabels(['TD3','SAC','PPO'])#

show()