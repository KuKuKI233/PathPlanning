from cmath import sqrt
from importlib.resources import path
import os
import sys
import math
import heapq
from xmlrpc.client import TRANSPORT_ERROR
import numpy as np

import matplotlib.pyplot as plt
from math import pi, sin, cos, acos, atan2, sqrt


import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from HybridAstar import plotting, env

"""
constraint Astar注意事项：
1.节点生长步长太短会导致节点树无法分枝，同样，栅格分辨率太大也会导致节点树无法分支；
    建议节点生长步长>=1.414*栅格分辨率
2.未进行路径优化、加入损失函数（Voronoi势场函数）；

"""


"""
AStar流程
1.初始化，start加入OPEN，令其f=g+h=0,end加入OPEN，g=inf；
2.从OPEN中找到f最小的点s，（s是否为end，若为end则结束）加入CLOSE，寻找s的neighbor，计算其f；
                    其中，若s的neighbor为在障碍物中或s到neighber发生碰撞，将其成本g设为inf；
3.若neighbor在OPEN中，则计算从s到neighbor的成本是否小于neighber的原成本
                    若小于，则修正s为neighber的父节点；
                    若不小于，则不做操作；
  若neighbor不再OPEN中，则将其加入到OPEN中；
4.返回2
"""


class ConstraintAStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, direc_start,s_goal, direc_goal,heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.direc_start = direc_start
        self.direc_goal = direc_goal
        self.heuristic_type = heuristic_type
        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.weight = 1 #启发式的权重
        self.path=[]
        self.direction=dict()
        self.motion_mode=dict() #该节点被扩展的方式： 1 左转，0 直行，-1 右转；
        self.path_end=s_start

        self.dubins_path_part=[]

        #设置小车的属性
        self.car_length=3
        self.car_width=1.5
        self.car_wheel_tread=2.5 #求前后轮距
        self.car_dist_rearwheel2rear=0.5 #后轮到后部的距离
        self.car_max_steer_angle=pi/6 #前轮最大转向角
        #最小转弯半径
        self.car_min_turnning_radius=self.car_wheel_tread/(math.tan(self.car_max_steer_angle))
        self.car_node_steer_radius=1.8#节点扩展长度

        #计算节点扩展属性
        #扩展的子节点坐标与父节点坐标的偏差角度
        self.node_max_steer_angle=math.in(0.5*self.car_node_steer_radius/self.car_min_turnning_radius)
        #扩展的子节点方向与的父节点方向的偏差角度
        self.node_yaw_angle=2*self.node_max_steer_angle
        # print(self.node_max_steer_angle/3.14)
        #设置最大dubins路径长度
        self.max_dubins_path_length=self.car_min_turnning_radius*4
        
    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.PARENT[self.s_start] = self.s_start
        self.direction[self.s_start]=self.direc_start
        self.direction[self.s_goal]=self.direc_goal
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        #以f值为堆索引 堆（索引，坐标）
        heapq.heappush(self.OPEN,(self.f_value(self.s_start), self.s_start))
        times=0
        while self.OPEN:
            times=times+1
            _, s = heapq.heappop(self.OPEN) #弹出索引最小值
            # print("min s=",self.g[s])
            self.CLOSED.append(s)
            
            if(self.use_dubins_curves_arrive_goal(s)):  # stop condition
                # s_final=(round(s[0]),round(s[1]))
                # print(self.dubins_path_part)
                print("end:",self.dubins_path_part[-1])
                self.path_end=s
                print("arrive goal",s)
                break
            
            # for s_n in self.get_neighbor(s):
            x=0
            for s_n in self.expanding_node(s):
                x=x+1
                # print(x)
                #若s至s_n发生碰撞，则cost=inf
                new_cost = self.g[s] + self.cost(s, s_n)

                #若s_n 不在OPEN列表中，则其没有g值，将其g设为inf，便于下一步将该点加入OPEN中
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost

                    
                    if(self.h1(s_n)>=self.max_dubins_path_length):
                        h1=self.max_dubins_path_length
                    else:
                        h1=self.h1(s_n)
                    h2=self.h2(s_n)
                    #转向角度惩罚0.5，子节点方向与父节点方向不同则增加成本；
                    if(self.direction[s_n]!=self.direction[self.PARENT[s_n]]):
                        h2=h2+0.5

                    # if (times%20==0):
                    #     print("h=",self.h1(s_n),h1,h2,"s_n",s_n)

                    heapq.heappush(self.OPEN, (max(h1,h2), s_n))
        # if (times%200==0):
        #     print("times=",times)
        return self.extract_path(self.PARENT), self.CLOSED
    
    def Astar_searching(self,s):
        start=(round(s[0]),round(s[1]))
        A_parent=dict()
        A_parent[start]=start
        A_g=dict()
        
        A_g[start]=0
        A_g[self.s_goal]=math.inf
        #以f值为堆索引 堆（索引，坐标）
        open = []
        close = []
        heapq.heappush(open,(self.A_f(start,A_g[start]), start))
        n=0
        # print(open)
        while open:
            n=n+1
            # print(n)
            _, s = heapq.heappop(open) #弹出索引最小值
            close.append(s)
            if s == self.s_goal:  # stop condition
                # print(self.s_goal)
                # print("s=goal")
                break
            
            for s_n in self.get_neighbor(s):
                
                #若s至s_n发生碰撞，则cost=inf
                new_cost = A_g[s] + self.cost(s, s_n)
                if s_n not in A_g:
                    A_g[s_n] = math.inf
                if new_cost < A_g[s_n]:  # conditions for updating Cost
                    A_g[s_n] = new_cost
                    A_parent[s_n]=s
                    heapq.heappush(open, (self.A_f(s_n,A_g[s_n]), s_n))
        
        h2=A_g[A_parent[self.s_goal]]
        # print("h2=",h2)
        # self.extract_path(A_parent)
        return h2

    def A_f(self,s,A_g_s):
        return A_g_s+self.weight*self.A_h(s)

    def A_h(self,s):
        heuristic_type = self.heuristic_type  # heuristic type
        #goal = self.s_goal  # goal node
        if heuristic_type == "manhattan":
            return abs(self.s_goal[1]-s[1])+abs(self.s_goal[0]-s[0])
        else:
            return math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1]) 
    
    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def expanding_node(self,s):
        # dirc=self.direction[s]
        R=self.car_node_steer_radius
        # R=0.2
        expanding_node=[]
        #扩展三个方向上的节点：
        #左转
        e_n=(s[0]+R*math.cos(self.direction[s]+1*self.node_yaw_angle),s[1]+R*math.sin(self.direction[s]+1*self.node_yaw_angle))
        expanding_node.append(e_n)
        self.PARENT[e_n]=s
        self.motion_mode[e_n]=1
        self.get_node_direction(e_n)
        #直行
        e_n=(s[0]+R*math.cos(self.direction[s]+0*self.node_yaw_angle),s[1]+R*math.sin(self.direction[s]+0*self.node_yaw_angle))
        expanding_node.append(e_n)
        self.PARENT[e_n]=s
        self.motion_mode[e_n]=0
        self.get_node_direction(e_n)
        #右转
        e_n=(s[0]+R*math.cos(self.direction[s]-1*self.node_yaw_angle),s[1]+R*math.sin(self.direction[s]-1*self.node_yaw_angle))
        expanding_node.append(e_n)
        self.PARENT[e_n]=s
        self.motion_mode[e_n]=-1
        self.get_node_direction(e_n)

        return expanding_node

    def get_node_direction(self,s):
        
        self.direction[s]=self.direction[self.PARENT[s]]+self.motion_mode[s]*self.node_yaw_angle
        
        return True

    def f_value(self,s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        return self.g[s]+self.weight*self.heuristic(s)

        # return max(self.h1(s),self.h2(s))
    
    def heuristic(self,s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """
        heuristic_type = self.heuristic_type  # heuristic type
        #goal = self.s_goal  # goal node
        if heuristic_type == "manhattan":
            return abs(self.s_goal[1]-s[1])+abs(self.s_goal[0]-s[0])
        else:
            return math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])

    def h1(self,s):
        """无障碍物的非完整约束启发代价"""
        # self.direction[self.s_start]=self.direc_start
        # self.direction[self.s_goal]=self.direc_goal
        # print(s)
        # print(self.direction[s])
        # print(self.s_goal)
        # print(self.direction[self.s_goal])

        L,ind =dubins_L(s,self.direction[s],self.s_goal,self.direction[self.s_goal],self.car_min_turnning_radius)
        
        return L[ind][0]

    def h2(self,s):
        """有障碍物的完整性启发式代价
            只考虑障碍物信息而不考虑车辆的非完整性约束条件
            （优点是引入该启发函数后能够发现2D空间中所有的U形障碍物和死胡同dead end）。
            随后使用2D动态规划的方法（其实就是传统的2D A* 算法）计算每个节点到终点的最短路径。
        """
        return self.Astar_searching(s)

    def get_dubins_curves(self,s):
        dx=self.s_goal[0]-s[0]
        dy=self.s_goal[1]-s[1]
        d=sqrt(dx*dx+dy*dy)/self.car_min_turnning_radius


        return True

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        # if self.is_collision(s_start, s_goal):
        #     return math.inf
        if self.is_collide_with_obs(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False
    
    def is_collide_with_obs(self, float_start, float_end):
        #简化，对质点坐标取整,只检查质点是否在障碍物中；
        s_start=(round(float_start[0]),round(float_start[1]))
        s_end=(round(float_end[0]),round(float_end[1]))
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True
                
        return False
    
     
    def is_line_collision(self, s_start, s_end):
        """计算直线是否与障碍物发生碰撞
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """  
        # print(s_end[1]-s_start[1],(s_end[0]-s_start[0]))
        k=(s_end[1]-s_start[1])/(s_end[0]-s_start[0])
        # print("k",k)
        if(k<=1 and k>=-1):
            if((s_end[0]-s_start[0]))>=0:
                sign=1
            else:
                sign=-1
            for i in range(0,(s_end[0]-s_start[0])*sign):
                # print(i)
                if(i==0):
                    intersection_x=s_start[0]+0.5*sign
                else:
                    intersection_x=intersection_x+1*sign
                # print("intersection_x",intersection_x)
                intersection_y= k * (intersection_x - s_start[0]) + s_start[1]
                # print(s_start[0],s_start[1])
                # print("intersection_y",intersection_y)
                x1=math.floor(intersection_x)
                x2=x1+1
                y0=math.floor(intersection_y)
                if ((intersection_y - y0) < 0.5):
                    # print("< 0.5")
                    y1 = y0
                    s1=(x1,y1)
                    s2=(x2,y1)
                    # print(s1,s2)
                    if(s1 in self.obs or s2 in self.obs):
                        return True
                elif ((intersection_y - y0) == 0.5):
                    # print("= 0.5")
                    y1 = y0; y2 = y0 + 1
                    s1=(x1,y1)
                    s2=(x2,y1)
                    s3=(x1,y2)
                    s4=(x2,y2)
                    # print(s1,s2,s3,s4)
                    if(s1 in self.obs or s2 in self.obs or s3 in self.obs or s4 in self.obs):
                        return True
                else:
                    # print("> 0.5")
                    y1 = y0 + 1
                    s1=(x1,y1)
                    s2=(x2,y1)
                    # print(s1,s2)
                    if(s1 in self.obs or s2 in self.obs):
                        return True        
        else:
            if((s_end[1]-s_start[1]))>=0:
                sign=1
            else:
                sign=-1
            for i in range(0,(s_end[1]-s_start[1])*sign):
                if(i==0):
                    intersection_y=s_start[1]+0.5*sign
                else:
                    intersection_y=intersection_y+1*sign
                intersection_x = (intersection_y - s_start[1])/k + s_start[0]

                y1=math.floor(intersection_y)
                y2 = y1 + 1
                x0=math.floor(intersection_x)
                if ((intersection_x - x0) < 0.5):
                    x1 = x0
                    if((x1,y1) in self.obs or (x1,y2) in self.obs):
                        return True
                elif ((intersection_x - x0) == 0.5):
                    x1 = x0; x2 = x0 + 1
                    if((x1,y1) in self.obs or (x2,y1) in self.obs or (x1,y2) in self.obs or (x2,y2) in self.obs):
                        return True
                else:
                    x1 = x0 + 1
                    if((x1,y1) in self.obs or (x1,y2) in self.obs):
                        return True
        return False

    def is_arrive_goal(self,s):
        s_final=(round(s[0]),round(s[1]))
        if(s_final==self.s_goal):
            return True
        return False

    def use_dubins_curves_arrive_goal(self,s):
        L,ind=dubins_L(s,self.direction[s],self.s_goal,self.direction[self.s_goal],self.car_min_turnning_radius)
        path_length=L[ind][0]

        if path_length>self.max_dubins_path_length: 
            self.dubins_path_part=[]
            return False
        else:
            _,d_path,path_points=get_dubins_path(s,self.direction[s],self.s_goal,self.direction[self.s_goal],self.car_min_turnning_radius,L,ind)
            
            for j in range(0,int(np.array(d_path).size/3)):
                
                pi=(round(d_path[j][0]),round(d_path[j][1]))
                if pi in self.obs:
                    print("dubins path collision")
                    self.dubins_path_part=[]
                    return False
            print("get dubins path ")
            if(sqrt((path_points[-1][0]-self.s_goal[0])**2+(path_points[-1][1]-self.s_goal[1])**2)>1):
                return False
            self.dubins_path_part=path_points
            # print(path_points)
            return True

    def extract_path(self, PARENT,):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        # print("xx")
        # self.path = [self.path_end]
        # s = self.path_end
        self.path = [self.path_end]
        s = self.path_end
        
        while True:
            # time.sleep(0.1)
            # print(s)
            s = PARENT[s]
            self.path.append(s)

            if s == self.s_start:
                break
        
        self.path.reverse()
        # print(np.array(self.dubins_path_part).size)
        for i in range(0,int(np.array(self.dubins_path_part).size/2)):
            self.path.append(self.dubins_path_part[i])

        # self.path=self.delete_colilinear_points(self.path)       
        # self.path=self.delete_redundant_inflection_points(self.path)
        # print(list(self.path))
        return list(self.path)


def dubins_L(s_start,d_start,s_goal,d_goal,r):
    dx=s_goal[0]-s_start[0]
    dy=s_goal[1]-s_start[1]
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
    ind=np.argmin(L,0)[0] #选择路径长度最短的情况

    return L,ind 

def get_dubins_path(s_start,d_start,s_goal,d_goal,r,L,ind):
    """
    return:path list, path2 np.arrat
    """
    types=['LSL','LSR','RSL','RSR','RLR','LRL']
    # 坐标变换后的起点
    p_start=[0,0,d_start]    
    mid1=dubins_segment(L[ind][1],p_start,types[ind][0])#第一段的运动路径结束点
    mid2=dubins_segment(L[ind][2],mid1,types[ind][1])#第二段的运动路径结束点
    
    path=[]
    path_points=[]
    # path_point=[0,0]
    #对dubins路径进行离散
    interval_step=0.1 #离散路径步长（原坐标系）
    num_step=int((L[ind][0]*r)//interval_step)#路径点个数
    
    for i in range(0,num_step+2):
        step=i*interval_step #离散路径步长，转换后坐标系中
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
        path_point=(end_pt[0],end_pt[1])
        path_points.append(path_point)
        # print(end_pt)
    path2=np.array(path)
    # print(path2[:,0])
    
    return path,path2,path_points
    
#dubins曲线计算部分
def dubins_path(s_start,d_start,s_goal,d_goal,r):
    """
    计算dubins曲线
    :param:s_start,d_start,s_goal,d_goal: 起始点、目标点
           r:最小转弯半径
    :return: path2 dubins路径(np.array类型)
    """
    dx=s_goal[0]-s_start[0]
    dy=s_goal[1]-s_start[1]
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
    path_points=[]
    path_point=[]
    #对dubins路径进行离散
    interval_step=0.1 #离散路径步长（原坐标系）
    num_step=int((L[ind][0]*r)//interval_step)#路径点个数
    
    for i in range(0,num_step+2):
        step=i*interval_step #离散路径步长，转换后坐标系中
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
        path_point[0]=end_pt[0]
        path_point[1]=end_pt[1]
        path_points.append(path_point)
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


def main():
    s_start = (10, 70) #坐标
    s_goal = (95, 60)
    direc_start=0*math.pi/2 #方向
    direc_goal=1*math.pi/2
    # s_start = (2, 2)
    # s_goal = (38, 2)
    

    astar = ConstraintAStar(s_start, direc_start,s_goal,direc_goal ,"euclidean") #euclidean manhattan
    # print("is line collision? ",astar.is_line_collision(s_start,s_goal))

    plot = plotting.Plotting(s_start, s_goal)
    path, visited = astar.searching()
    plot.animation(path, visited, "constraint A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")

    # print(astar.h1(s_start))
if __name__ == '__main__':
    
    main()
