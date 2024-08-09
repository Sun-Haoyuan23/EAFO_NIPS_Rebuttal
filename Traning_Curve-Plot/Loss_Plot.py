import numpy as np
# open the file
f_CRReLU=open('./experiments_300epoch/cifar100/convnext_large/crrelu/log.txt', encoding='gbk')
txt_CRReLU=[]
for line in f_CRReLU:
    txt_CRReLU.append(eval(line.strip()))

acc_1_CRReLU=[]

# get the top1 accuracy list

for i in range(len(txt_CRReLU)):
    a = txt_CRReLU[i]
    b=a['test_loss']
    c=b
    acc_1_CRReLU.append(c)


f_gelu=open('./experiments_300epoch/cifar100/convnext_large/gelu/log.txt', encoding='gbk')
txt_gelu=[]
for line in f_gelu:
    txt_gelu.append(eval(line.strip()))

acc_1_gelu=[]


for i in range(len(txt_gelu)):
    a = txt_gelu[i]
    b=a['test_loss']
    c=b
    acc_1_gelu.append(c)
y_gelu=np.array(acc_1_gelu)


f_elu=open('./experiments_300epoch/cifar100/convnext_large/elu/log.txt', encoding='gbk')
txt_elu=[]
for line in f_elu:
    txt_elu.append(eval(line.strip()))

acc_1_elu=[]


for i in range(len(txt_elu)):
    a = txt_elu[i]
    b=a['test_loss']
    c=b
    acc_1_elu.append(c)
y_elu=np.array(acc_1_elu)

f_elu=open('./experiments_300epoch/cifar100/convnext_large/elu/log.txt', encoding='gbk')
txt_elu=[]
for line in f_elu:
    txt_elu.append(eval(line.strip()))

acc_1_elu=[]


for i in range(len(txt_elu)):
    a = txt_elu[i]
    b=a['test_loss']
    c=b
    acc_1_elu.append(c)
y_elu=np.array(acc_1_elu)

f_prelu=open('./experiments_300epoch/cifar100/convnext_large/prelu/log.txt', encoding='gbk')
txt_prelu=[]
for line in f_prelu:
    txt_prelu.append(eval(line.strip()))

acc_1_prelu=[]


for i in range(len(txt_prelu)):
    a = txt_prelu[i]
    b=a['test_loss']
    c=b
    acc_1_prelu.append(c)
y_prelu=np.array(acc_1_prelu)

f_mish=open('./experiments_300epoch/cifar100/convnext_large/mish/log.txt', encoding='gbk')
txt_mish=[]
for line in f_mish:
    txt_mish.append(eval(line.strip()))

acc_1_mish=[]


for i in range(len(txt_mish)):
    a = txt_mish[i]
    b=a['test_loss']
    c=b
    acc_1_mish.append(c)
y_mish=np.array(acc_1_mish)

f_silu=open('./experiments_300epoch/cifar100/convnext_large/silu/log.txt', encoding='gbk')
txt_silu=[]
for line in f_silu:
    txt_silu.append(eval(line.strip()))

acc_1_silu=[]


for i in range(len(txt_silu)):
    a = txt_silu[i]
    b=a['test_loss']
    c=b
    acc_1_silu.append(c)
y_silu=np.array(acc_1_silu)

f_starrelu=open('./experiments_300epoch/cifar100/convnext_large/starrelu/log.txt', encoding='gbk')
txt_starrelu=[]
for line in f_starrelu:
    txt_starrelu.append(eval(line.strip()))

acc_1_starrelu=[]


for i in range(len(txt_starrelu)):
    a = txt_starrelu[i]
    b=a['test_loss']
    c=b
    acc_1_starrelu.append(c)
y_starrelu=np.array(acc_1_starrelu)


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

x=[]
for i in range(len(acc_1_CRReLU)):
    x.append(i+1)
x=np.array(x)
y_CRReLU=np.array(acc_1_CRReLU)

# get the Line chart plot

import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.figure(figsize=(6, 4))
plt.plot(x,y_gelu,color = 'purple',label="GELU", alpha=0.8,linewidth=0.8)
plt.plot(x,y_prelu,color = 'dodgerblue',label="PReLU", alpha=0.8,linewidth=0.8)
plt.plot(x,y_elu,color = 'darkorange',label="ELU", alpha=0.8,linewidth=0.8)
plt.plot(x,y_mish,color = 'gray',label="Mish", alpha=0.8,linewidth=0.8)
plt.plot(x,y_silu,color = 'aqua',label="SiLU", alpha=0.8,linewidth=0.8)
plt.plot(x,y_starrelu,color = 'limegreen',label="StarReLU", alpha=0.8,linewidth=0.8)
plt.plot(x,y_CRReLU,color = 'r',label="CRReLU", alpha=0.9,linewidth=0.8)
plt.xlabel("Epoch", fontproperties = "Times New Roman", fontsize=12)#横坐标名字
plt.ylabel("Test Loss",fontproperties = "Times New Roman",fontsize=12)#纵坐标名字
plt.tick_params(labelsize=8)
plt.legend(loc = "upper right",prop = {'size':8, "family":'Times New Roman'})#图例
plt.savefig("./convnext_large-cifar100_loss.jpg", dpi=1200)