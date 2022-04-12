"""
utils for collision check
@author: huiming zhou
"""

from cmath import sqrt
import math
from turtle import shape
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from MyRRT_2D import env
from MyRRT_2D.rrt import Node


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        #不考虑重合、平行
        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        # o为起点，d为方向、距离
        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        # v1, v2, v3, v4为矩形四个顶点
        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False

    def is_collide_with_obs(self,node):
        delta = self.delta
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True
        for (x, y, w, h) in self.obs_rectangle:
            # min_x=x-(w/2);max_x=x+(w/2)
            # min_y=y-(h/2);max_y=y+(h/2)
            min_x=x;max_x=x+(w)
            min_y=y;max_y=y+(h)
            #print(min_x,max_x,min_y,max_y)
            dist_x=max(min_x-node.x,0,node.x-max_x)
            dist_y=max(min_y-node.y,0,node.y-max_y)
            #print(dist_x,dist_y)
            if dist_x==0 and dist_y==0:
                return True
            else:
                dist=abs(sqrt((dist_x)*(dist_x)+(dist_y)*(dist_y)))
                #print(dist)
                if dist<= delta:                    
                    return True
        return False 
                    
    def cal_dist_node_obs(self, node):
        #dist_node_and_obs=np.array([])
        #初始化node与cir、rec的距离
        n_circle=(np.array(self.obs_circle).shape)[0]
        dist_node_and_cir=[[0.0,0.0,0.0]]*n_circle
        dist_node_and_cir=np.array(dist_node_and_cir)

        n_rectangle=(np.array(self.obs_rectangle).shape)[0]
        dist_node_and_rec=[[0.0,0.0,0.0]]*n_rectangle
        dist_node_and_rec=np.array(dist_node_and_rec)
        delta = self.delta
        dist_node_and_obs=[[0.0,0.0,0.0]]*(n_rectangle+n_circle)
        dist_node_and_obs=np.array(dist_node_and_obs)
        n1=0
        n2=0
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                #node inside cir时，dist=0
                dist_node_and_obs[n1][0]=0
                dist_node_and_obs[n1][1]=0
                dist_node_and_obs[n1][2]=0
            else:
                #dist_node_and_cir[n1][0] node与cir的距离，dist_node_and_cir[n1][1]为cir到node的方向x
                dist=math.hypot(node.x - x, node.y - y) - r - delta
                dist_x=(node.x - x)/(math.hypot(node.x - x, node.y - y)) #归1化的x方向
                dist_y=(node.y - y)/(math.hypot(node.x - x, node.y - y)) #归1化的y方向
                dist_node_and_obs[n1][0]=dist
                dist_node_and_obs[n1][1]=dist_x
                dist_node_and_obs[n1][2]=dist_y
            n1=n1+1
        #dist_node_and_obs=dist_node_and_cir 
        #求rec至node的最近距离以及该距离的方向（rec至node）
        for (x, y, w, h) in self.obs_rectangle: 
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                dist_node_and_obs[n1+n2][0]=0
                dist_node_and_obs[n1+n2][1]=0
                dist_node_and_obs[n1+n2][2]=0
            else:
                min_x=x;max_x=x+(w)
                min_y=y;max_y=y+(h)
                dist_x=max(min_x-node.x,0,node.x-max_x)
                dist_y=max(min_y-node.y,0,node.y-max_y)
                dist=abs(sqrt((dist_x)*(dist_x)+(dist_y)*(dist_y)))
                #print(dist)
                dist_node_and_obs[n1+n2][0]=dist - delta
                dist_node_and_obs[n1+n2][1]=dist_x/(dist)
                dist_node_and_obs[n1+n2][2]=dist_y/(dist)
            n2=n2+1  
        return dist_node_and_obs

    # def cal_repul_force(self,node):
    #     bia_x=0;bia_y=0 #bia_x 计算得节点的偏差位移
    #     Dist_0=2 #排斥势场范围
    #     dist_node_and_obs=self.cal_dist_node_obs(node)
    #     max_n=dist_node_and_obs.shape[0]
    #     print(max_n)
    #     for n in range(0,max_n):
    #         if dist_node_and_obs[n][0]>0 and dist_node_and_obs[n][0]<1:
    #             Dist_b=dist_node_and_obs[n][0]
    #             bia_x=bia_x+0.2*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][1]
    #             bia_y=bia_y+0.2*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][2]
    #     bia=[bia_x,bia_y]
    #     return bia


    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)

    #新增
    #计算障碍物与node的距离
    def get_dist_node_and_obs(self, node):
        #1.暂时只考虑圆形障碍物
        #n_circle为circle个数
        n=0
        n_circle=(np.array(self.obs_circle).shape)[0]
        dist_node_and_obs=[0]*n_circle
        print(dist_node_and_obs)
        for (x,y,r) in self.obs_circle:            
            dist_node_and_obs[n]=math.hypot(node.x - x, node.y - y)-r
            n=n+1                          
        return dist_node_and_obs
