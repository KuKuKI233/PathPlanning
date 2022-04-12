"""
@author: xia
"""

import os
import sys
import math
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from MyRRT_2D import env, plotting, utils, queue


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class APGDRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate,repul_bias_rate ,search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.repul_bias_rate = repul_bias_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.rt = 0.0
        self.path_cost =0.0
    def planning(self):
        #只有到达迭代次数才停止
        start_time = time.time()
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)
            if node_new and not self.utils.is_collide_with_obs(node_new):
                #print("not inside")
                node_new = self.bias_node_new(node_new)

            if k % 500 == 0:
                print(k)

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        if index:
            print("find path")
        self.path = self.extract_path(self.vertex[index])
        end_time = time.time()
        self.rt=end_time-start_time
        print("runningtime:" , self.rt)
        self.path_cost=self.cost(self.vertex[index])
        print("C=",self.path_cost)
        self.plotting.animation(self.vertex, self.path, "APGDrrt*, N = " + str(self.iter_max))
        
        

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    # node_new选择潜在父节点
    def choose_parent(self, node_new, neighbor_index):
        #计算 node_new 和 邻节点群中的各节点之间的成本；
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        #成本最小者为 node_new 的 暂时的父节点；
        node_new.parent = self.vertex[cost_min_index]

    # 重新布线
    # 在neighbor_index的节点i中，若以node_new为i的父亲节点时的成本小于原本i的成本，则重新选择以node_new为i的父节点
    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    #选择一条成本最小的路径，以该路径末端为目标点的父节点
    def search_goal_parent(self):
        #dist_list 为 所有节点与目标点的距离
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        #找到目标点附近一个步长范围内所有节点 为 node_index
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        #遍历node_index，找到成本最小的goal的父节点,该成本等与父节点路径成本+父节点至目标成本
        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    #计算node_start的成本 + node_start到node_end的成本
    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    #寻找node_new 的 最近邻节点群（节点列表），作为潜在父节点
    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        #r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)
        r = max(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)
        #r=self.search_radius
        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        #若vertex中，与node_new距离小于r，且之间直连不发生碰撞，加入到临节点群dist_table_index中；
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        #计算node_p的成本
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def update_cost(self, parent_node):
        OPEN = queue.QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    #计算吸引势场方向，决定偏置的方向
    def find_att_field_direction(self,node):
        #self.s_goal
        dist= math.hypot(self.s_goal.x-node.x, self.s_goal.y-node.y)
        if (dist==0):
            #dist==0 即到达目标点
            dir=[0,0]
        else:
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
        dir=self.find_att_field_direction(node)
        rep_bias= self.cal_rep_bias_dist(node)
        node.x=dir[0]*(0.5-rep_bias)+node.x
        node.y=dir[1]*(0.5-rep_bias)+node.y
        return node

    #依概率使用cal_node_after_bias
    def bias_node_new(self, node_new):
        if np.random.random() < self.repul_bias_rate:
            bias_node = self.cal_node_after_bias(node_new)
        else:
            bias_node = node_new
        return bias_node


def main():
    x_start = (2, 2)  # Starting node
    x_goal = (48, 28)  # Goal node

    #x_start, x_goal, step_len, goal_sample_rate,repul_bias_rate ,search_radius, iter_max
    APGD_rrt_star = APGDRrtStar(x_start, x_goal, 5, 0.0, 0.0 ,1, 1000)
    APGD_rrt_star.planning()


if __name__ == '__main__':
    main()
