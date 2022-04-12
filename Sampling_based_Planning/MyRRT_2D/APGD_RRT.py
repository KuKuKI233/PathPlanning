"""
APGD_RRT
"""

"""
吸引势决定向目标的偏置方向
排斥势决定向目标偏置的步长（也可能导致远离目标）
"""
import os
import sys
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from MyRRT_2D import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, repul_bias_rate,iter_max):
        """s_start:起始点坐标，s_goal:终点坐标，step_len:步长，goal_sample_rate:向目标采样率
            iter_max:最大采样次数（iter:通路）
        """
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start] #存放节点
        self.repul_bias_rate = repul_bias_rate #偏执概率

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils() #存放工具类函数;常用工具

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.num_node=0

    def planning(self):
        for i in range(self.iter_max):
            # if i%100==0:
            #     print(i)

            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)
            #print(node_new.x,node_new.y)
            if node_new and not self.utils.is_collide_with_obs(node_new):
                #print("not inside")
                node_new = self.bias_node_new(node_new)

            #若生成node_new，node_new与树中最近的节点node_near无碰撞，则将node_new添加到vertex
            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)
                
                #是否到达目标点
                if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    return self.extract_path(node_new)
            self.num_node=i
        print("generate nodes:",self.num_node)
        return None

    #随机采样
    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            #在边界内部随机采样，生成Node
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def bias_node_new(self, node_new):
        if np.random.random() < self.repul_bias_rate:
            bias_node = self.cal_node_after_bias(node_new)
        else:
            bias_node = node_new
        return bias_node

    @staticmethod
    def nearest_neighbor(node_list, n):
        #下面用到for的高级用法：  [x**2 for x in L] ，其中x**2是使用到x的元素的算式，为return
        # np.argmin 返回输入list中最小元素的下标（下标从0开始）
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y) 
                                        for nd in node_list]))]

    #树向随机采样的节点生长一个节点node_new，并返回该node_new
    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    #计算排斥力
    def cal_rep_force(self,node):
        bias_x=0;bias_y=0 #bia_x 计算得节点的偏差位移
        Dist_0=2 #排斥势场范围
        dist_node_and_obs= utils.Utils().cal_dist_node_obs(node)
        max_n=dist_node_and_obs.shape[0]
        #print(max_n)
        for n in range(0,max_n):
            if dist_node_and_obs[n][0]>0 and dist_node_and_obs[n][0]<1:
                Dist_b=dist_node_and_obs[n][0]
                # bias_x=bias_x+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][1]
                # bias_y=bias_y+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][2]
                bias_x=bias_x+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][1]
                bias_y=bias_y+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][2]
        bias=[bias_x,bias_y]
        if (math.hypot(bias_x, bias_y)>=1):
            bias_x=(1*bias_x)/(math.hypot(bias_x, bias_y))
            bias_y=(1*bias_y)/(math.hypot(bias_x, bias_y))
            bias=[bias_x,bias_y]
        return bias

    def cal_att_force(self,node):
        bias_x=0;bias_y=0 #bia_x 计算得节点的偏差位移
        # self.s_goal #目标坐标
        dist= math.hypot(self.s_goal.x-node.x, self.s_goal.y-node.y)
        if dist<=5:
            bias_x=0.2*(self.s_goal.x-node.x)
            bias_y=0.2*(self.s_goal.y-node.y)
        else:
            bias_x=(self.s_goal.x-node.x)/dist
            bias_y=(self.s_goal.y-node.y)/dist
        bias=[bias_x,bias_y]
        return bias

    # #计算经过吸引力、排斥力偏置后的node
    # def cal_node_after_bias(self,node):
    #     bias_att = Node(self.cal_att_force(node))
    #     bias_rep = Node(self.cal_rep_force(node))
    #     node.x=node.x+bias_att.x+bias_rep.x
    #     node.y=node.y+bias_att.y+bias_rep.y
    #     return node

    #计算吸引势场方向，决定偏置的方向
    def find_att_field_direction(self,node):
        #self.s_goal
        dist= math.hypot(self.s_goal.x-node.x, self.s_goal.y-node.y)
        dir_x = (self.s_goal.x-node.x)/dist
        dir_y = (self.s_goal.y-node.y)/dist
        dir=[dir_x,dir_y]
        return dir
    #计算排斥偏执距离
    def cal_rep_bias_dist(self,node):
        bias_x=0;bias_y=0 #bia_x 计算得节点的偏差位移
        Dist_0=2 #排斥势场范围
        dist_node_and_obs= utils.Utils().cal_dist_node_obs(node)
        max_n=dist_node_and_obs.shape[0]
        #print(max_n)
        for n in range(0,max_n):
            if dist_node_and_obs[n][0]>0 and dist_node_and_obs[n][0]<1:
                Dist_b=dist_node_and_obs[n][0]
                bias_x=bias_x+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][1]
                bias_y=bias_y+0.0002*((1/Dist_b)-(1/Dist_0))*((1/Dist_b)*(1/Dist_b))*dist_node_and_obs[n][2]
        bias=math.hypot(bias_x, bias_y)
        if (bias>=1):
            bias=1
        return bias

    #计算APGD偏置后的节点
    def cal_node_after_bias(self,node):
        # bias_att = Node(self.cal_att_force(node))
        # bias_rep = Node(self.cal_rep_force(node))
        # node.x=node.x+bias_att.x+bias_rep.x
        # node.y=node.y+bias_att.y+bias_rep.y
        dir=self.find_att_field_direction(node)
        rep_bias= self.cal_rep_bias_dist(node)
        #默吸引势认向目标偏置0.5，排斥势最大向远离目标偏执1
        node.x=dir[0]*(0.5-rep_bias)+node.x
        node.y=dir[1]*(0.5-rep_bias)+node.y

        return node


    #生成path
    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))
        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node
    # s_start, s_goal, step_len, goal_sample_rate, repul_bias_rate,iter_max
    rrt = Rrt(x_start, x_goal, 0.5, 0.3,0.3,20000)
    path = rrt.planning()

    if path:
        print("numbers of node:",rrt.num_node)
        rrt.plotting.animation(rrt.vertex, path, "APGD_RRT", True)
        #print()
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()
