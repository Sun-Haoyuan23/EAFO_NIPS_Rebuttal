import numpy as np
# open the file
f_CRReLU=open('./experiments_300epoch/cifar100/vit_tiny_patch4_32/crrelu/log.txt', encoding='gbk')
txt_CRReLU=[]
for line in f_CRReLU:
    txt_CRReLU.append(eval(line.strip()))

epsilon=[]

# get the top1 accuracy list

for i in range(len(txt_CRReLU)):
    a = txt_CRReLU[i]
    b=a['epsilon']
    c=b[0]
    epsilon.append(c)


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

x=[]
for i in range(300):
    x.append(i+1)
x=np.array(x)
y_CRReLU=np.array(epsilon)
print(epsilon)

# get the Line chart plot

import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.figure(figsize=(6, 4))
plt.plot(x,y_CRReLU,color = 'r',label="CRReLU", alpha=0.9,linewidth=0.8)
plt.xlabel("Epoch", fontproperties = "Times New Roman", fontsize=12)#横坐标名字
plt.ylabel("Epsilon",fontproperties = "Times New Roman",fontsize=12)#纵坐标名字
plt.tick_params(labelsize=8)
plt.legend(loc = "lower right",prop = {'size':8, "family":'Times New Roman'})#图例
plt.savefig("./EAFO-for-parameter-plot.jpg", dpi=1200)