import os
import struct
import sys
import math
import numpy as np
import heapq
# import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from MySearch_2D import plotting, env


class Waypoint:
    def __init__(self, n):
        self.coordinate = n
        self.x = n[0]
        self.y = n[1]
        self.neighborWaypoints=[]
        self.neighborWaypoints_dist=[]
    


class PRM:
    def __init__(self,s_start, s_goal, cut_distance, sample_num):
        self.s_start = s_start
        self.s_goal = s_goal
        self.cut_distance = cut_distance
        self.sample_num = sample_num

        self.env = env.Env()
        self.Env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)

        # self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range


        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.weight = 1 #启发式的权重
        self.waypoint_list=[]

    def making_plan(self):
        waypoint_list=self.sample_waypoints()
        print(waypoint_list)
        self.connect_watpoints(waypoint_list)
        self.waypoint_list = waypoint_list
        
        return self.astar_searching(waypoint_list)

    def sample_waypoints(self):
        """
        sample waypoint 
        """
        waypoint_list = []
        wp0=Waypoint(self.s_start)
        waypoint_list.append(wp0)

        for i in range(0,self.sample_num):

            wp=Waypoint((np.random.uniform(1, self.x_range), np.random.uniform(0, self.y_range)))
            waypoint_list.append(wp)

        wp1=Waypoint(self.s_goal)
        waypoint_list.append(wp1)
        
        return waypoint_list


    
    def connect_watpoints(self,waypoint_list):
        """ connect the neighbor waypoints """
        print(len(waypoint_list))
        for i in range(0,len(waypoint_list)):
            current_waypoint=waypoint_list[i]
            for j in range(0,len(waypoint_list)):
                if i==j:
                    continue
                neighbor_waypoint=waypoint_list[j]

                dis_x=abs(current_waypoint.coordinate[0]-neighbor_waypoint.coordinate[0])
                dis_y=abs(current_waypoint.coordinate[1]-neighbor_waypoint.coordinate[1])
                dist=math.sqrt(dis_x*dis_x+dis_y*dis_y)
                if(dist<self.cut_distance):
                    
                    # 加入邻节点序号、距离
                    waypoint_list[i].neighborWaypoints.append(j)
                    waypoint_list[i].neighborWaypoints_dist.append(dist)


    def astar_searching(self, waypoint_list):
        """
        search a path by using astar 
        """
        # 起点加入到OPEN队列
        self.PARENT[0] = 0
        self.g[0] = 0
        self.g[(len(waypoint_list)-1)] = math.inf

        find_path_flag=False

        heapq.heappush(self.OPEN,(self.f_value(0), 0))
        while self.OPEN:
            _, s_index = heapq.heappop(self.OPEN) #弹出索引最小值
            
            self.CLOSED.append(s_index)
            print("s_index,f",s_index,_)
            if(s_index==(len(waypoint_list)-1)):
                print(len(self.PARENT))
                print(self.PARENT)
                
                # s_index 为 goal.index
                print("FIND A PATH")
                find_path_flag=True
                break
            
            for s_n_index in waypoint_list[s_index].neighborWaypoints:

                new_cost = self.g[s_index] + self.cost(waypoint_list[s_index].coordinate, waypoint_list[s_n_index].coordinate)
                
                if s_n_index not in self.g:
                    self.g[s_n_index] = math.inf

                if new_cost < self.g[s_n_index]:

                    print("s_n_index,g",s_n_index,new_cost)
                    self.g[s_n_index] = new_cost
                    self.PARENT[s_n_index] = s_index
                    heapq.heappush(self.OPEN, (self.f_value(s_n_index), s_n_index))
        
        path = []
        all_points = []
        if find_path_flag:
            path, all_points = self.extract_path(self.PARENT)
        return path, all_points
                    
    def extract_path(self, PARENT):

        path_index = [len(self.waypoint_list)-1]
        print(path_index)
        s_index = len(self.waypoint_list)-1
        path_index.append(s_index)
        print(PARENT)

        while True:
            s_index = PARENT[s_index]
            path_index.append(s_index)
        
            if s_index == 0:
                break
        
        all_points = []
        for i in range(0,len(self.waypoint_list)):
            all_points.append(self.waypoint_list[i].coordinate)


        path=[]
        for i in range(0,len(path_index)):
            ind = path_index[i]
            path.append(self.waypoint_list[ind].coordinate)

        return path, all_points


    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_line_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
    

    def f_value(self,s_index):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        print("xx",s_index)
        return self.g[s_index]+self.weight*self.heuristic(s_index)

    def heuristic(self,s_index):
        self.waypoint_list
        s=self.waypoint_list[s_index].coordinate
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """
        

        return math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])



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
            for i in range(0,math.ceil((s_end[0]-s_start[0])*sign)):
                # print(i)
                if(i==0):
                    intersection_x=s_start[0]+0.5*sign
                else:
                    intersection_x=intersection_x+1*sign
                # print("intersection_x",intersection_x)
                intersection_y= k * (intersection_x - s_start[0]) + s_start[1]

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
            for i in range(0,math.ceil((s_end[1]-s_start[1])*sign)):
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

    # def plot_waypoints(self, waypoint_list):
    #     for i in range(0,len(waypoint_list)):
    #         plt.plot(self.waypoint_list[i].coordinate[0],
    #                     self.waypoint_list[i].coordinate[1])
    #     plt.show()
            

def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node


    prm=PRM(x_start,x_goal,10,200)
    path, all_point = prm.making_plan()
    print(path)
    if path:
        plot = plotting.Plotting(x_start, x_goal)
        plot.animation(path, all_point, "PRM")


if __name__ == '__main__':
    main()