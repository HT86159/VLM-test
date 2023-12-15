import matplotlib.pyplot as plt
import numpy as np
import pdb;pdb.set_trace()
all_data=[np.random.normal(0,std,100) for std in range(1,4)]

#首先有图（fig），然后有轴（ax）
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))


bplot1=axes[0].boxplot(all_data,
                       vert=True,
                       patch_artist=True)


bplot2 = axes[1].boxplot(all_data,
                         notch=True,
                         vert=True, 
                         patch_artist=True)



#颜色填充
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# 加水平网格线
for ax in axes:
    ax.yaxis.grid(True) #在y轴上添加网格线
    ax.set_xticks([y+1 for y in range(len(all_data))] ) #指定x轴的轴刻度个数
    ## [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
    ax.set_xlabel('xlabel') #设置x轴名称
    ax.set_ylabel('ylabel') #设置y轴名称

# 添加刻度
# 添加刻度名称，我们需要使用 plt.setp() 函数：

# 加刻度名称
plt.setp(axes, xticks=[1,2,3],
         xticklabels=['x1', 'x2', 'x3'])
# 我们的刻度数是哪些，以及我们想要它添加的刻度标签是什么。
        
plt.savefig("box.png")
