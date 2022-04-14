# from os import path
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, acos, atan2, sqrt
import math

def dubins_path(s_start,d_start,s_goal,d_goal,r):
    """
    计算dubins曲线
    :param:s_start,d_start,s_goal,d_goal: 起始点、目标点
           r:最小转弯半径
    :return: path2 dubins路径(np.array类型)
    """
    dx=s_goal[0]-s_start[0]
    dy=s_goal[1]-s_start[0]
    d = sqrt( dx*dx + dy*dy ) / r

    theta = mod(atan2( dy, dx ), 2*pi); #起点和终点航向角度差
    alpha=mod(d_start-theta,2*pi) #坐标系变换后起点的方向角度
    beta=mod(d_goal-theta,2*pi) #坐标系变换后终点的方向角度
    L=[]   #存储计算的六种情况的路径
    L.append(LSL(alpha,beta,d))
    L.append(LSR(alpha,beta,d))
    L.append(RSL(alpha,beta,d))
    L.append(RSR(alpha,beta,d))
    L.append(RLR(alpha,beta,d))
    L.append(LRL(alpha,beta,d))
    # print(L)
    ind=np.argmin(L,0)[0] #选择路径长度最短的情况
    # print(ind)

    types=['LSL','LSR','RSL','RSR','RLR','LRL']
    # 坐标变换后的起点
    p_start=[0,0,d_start]    
    mid1=dubins_segment(L[ind][1],p_start,types[ind][0])#第一段的运动路径结束点
    mid2=dubins_segment(L[ind][2],mid1,types[ind][1])#第二段的运动路径结束点
    
    path=[]
    #对dubins路径进行离散
    interval_step=0.1 #离散路径步长
    num_step=int((L[ind][0]*r)//interval_step)#路径点个数
    
    for i in range(0,num_step+2):
        step=i*interval_step #离散路径步长转换至原坐标系
        t=step/r
        if(t<L[ind][1]):
            end_pt = dubins_segment( t, p_start,types[ind][0])
        elif(t<(L[ind][1]+L[ind][2])):
            end_pt = dubins_segment( t-L[ind][1],mid1,types[ind][1])
        else:
            end_pt = dubins_segment( t-L[ind][1]-L[ind][2],mid2,types[ind][2])
        end_pt[0]=end_pt[0]*r+s_start[0]
        end_pt[1]=end_pt[1]*r+s_start[1]
        end_pt[2]=mod(end_pt[2],2*pi)
        path.append(end_pt)
        # print(end_pt)
    path2=np.array(path)
    # print(path2[:,0])
    
    return path2

def dubins_segment(seg_param, seg_init, seg_type):
    """
    计算dubins曲线各部分
    :param:seg_param 线段参数,seg_init 线段起始点,seg_type 线段类型
    :return:seg_end 线段结束点坐标和方向
    """
    seg_end=[0,0,0]
    if( seg_type == 'L' ) :
        seg_end[0] = seg_init[0] + math.sin(seg_init[2]+seg_param) - math.sin(seg_init[2])
        seg_end[1] = seg_init[1] - math.cos(seg_init[2]+seg_param) + math.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif( seg_type == 'R' ):
        seg_end[0] = seg_init[0] - math.sin(seg_init[2]-seg_param) + math.sin(seg_init[2])
        seg_end[1] = seg_init[1] + math.cos(seg_init[2]-seg_param) - math.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif( seg_type == 'S' ): 
        seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]
    return seg_end

def LSL(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    tmp0 = d + sin(alpha) - sin(beta)
    p_squared = 2 + (d*d) -(2*cos(alpha - beta)) + (2*d*(sin(alpha) - sin(beta)))
    if( p_squared < 0 ):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        tmp1 = atan2( (cos(beta)-cos(alpha)), tmp0 )
        t = mod((-alpha + tmp1 ), 2*pi)
        p = sqrt( p_squared )
        q = mod((beta - tmp1 ), 2*pi)    
        L=[t+p+q,t,p,q]
    return L

def LRL(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    tmp_lrl = (6. - d*d + 2*cos(alpha - beta) + 2*d*(- sin(alpha) + sin(beta))) / 8
    if( abs(tmp_lrl) > 1):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        p=mod(( 2*pi - acos(tmp_lrl)),2*pi)
        t=(-alpha+atan2(-cos(alpha)+cos(alpha),d+sin(alpha)-sin(beta))+p/2)
        t=t%(2*pi)
        q=(beta%(2*pi))-alpha+((2*p)%(2*pi))
        L=[t+p+q,t,p,q]
    return L

def LSR(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    p_squared = -2 + (d*d) + (2*cos(alpha - beta)) + (2*d*(sin(alpha)+sin(beta)));
    if( p_squared < 0 ):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        p    = sqrt( p_squared )
        tmp2 = atan2( (-cos(alpha)-cos(beta)), (d+sin(alpha)+sin(beta)) ) - atan2(-2.0, p)
        t    = mod((-alpha + tmp2), 2*pi)
        q    = mod(( -mod((beta), 2*pi) + tmp2 ), 2*pi)
        L=[t+p+q,t,p,q]
    return L

def RSR(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    tmp0 = d-sin(alpha)+sin(beta)
    p_squared = 2 + (d*d) -(2*cos(alpha - beta)) + (2*d*(sin(beta)-sin(alpha)))
    if( p_squared < 0 ):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        tmp1 = atan2( (cos(alpha)-cos(beta)), tmp0 )
        t = mod(( alpha - tmp1 ), 2*pi)
        p = sqrt( p_squared )
        q = mod(( -beta + tmp1 ), 2*pi)
        L=[t+p+q,t,p,q]
    return L

def RLR(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    tmp_rlr = (6. - d*d + 2*cos(alpha - beta) + 2*d*(sin(alpha)-sin(beta))) / 8.
    if( abs(tmp_rlr) > 1):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        p = mod(( 2*pi - acos( tmp_rlr ) ), 2*pi)
        t = mod((alpha - atan2( cos(alpha)-cos(beta), d-sin(alpha)+sin(beta) ) + mod(p/2, 2*pi)), 2*pi)
        q = mod((alpha - beta - t + mod(p, 2*pi)), 2*pi)
        L=[t+p+q,t,p,q]
    return L

def RSL(alpha,beta,d):
    """
    :param:alpha 起点方向角度，beta 终点方向角度，d 起点和终点距离
    :return:L dubins路径
    """
    p_squared = (d*d) -2 + (2*cos(alpha - beta)) - (2*d*(sin(alpha)+sin(beta)))
    if( p_squared< 0 ):
        L=[math.inf,math.inf,math.inf,math.inf]
    else:
        p    = sqrt( p_squared )
        tmp2 = atan2( (cos(alpha)+cos(beta)), (d-sin(alpha)-sin(beta)) ) - atan2(2.0, p)
        t    = mod((alpha - tmp2), 2*pi)
        q    = mod((beta - tmp2), 2*pi)
        L=[t+p+q,t,p,q]
    return L

def plot_path(s_start,d_start,s_goal,d_goal,r,path2):
    """绘制路径"""
    arrow_length=1#方向向量长度    
    #起点向量
    plt.arrow(s_start[0],s_start[1],arrow_length * cos(d_start),arrow_length * sin(d_start),\
                width=0.01,length_includes_head=True,head_width=0.05,head_length=0.1,fc='b',ec='b')  
    #绘制路径点
    plt.plot(path2[:,0],path2[:,1])
    #终点向量
    plt.arrow(s_goal[0],s_goal[1],arrow_length * cos(d_goal),arrow_length * sin(d_goal),\
                width=0.01,length_includes_head=True,head_width=0.05,head_length=0.1,fc='b',ec='b')
    ax = plt.gca()  # 设置绘图坐标轴等比例
    ax.set_aspect(1)
    plt.grid(True)
    plt.show()

def mod(x,y):
    """计算x除y的余数"""
    return x%y


if __name__ == "__main__":
    # print(RSL(3*pi/2,pi/2,10))
    s_start=(10,10) #起始点坐标
    d_start=0*pi/2  #起始点方向角度
    s_goal=(15,20)  #目标点坐标
    d_goal=0*pi/2   #目标点方向角度
    r=2

    path=dubins_path(s_start,d_start,s_goal,d_goal,r)
    
    plot_path(s_start,d_start,s_goal,d_goal,r,path)
