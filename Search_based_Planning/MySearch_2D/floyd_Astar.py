import os
import sys
import math
import heapq
import numpy as np

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from MySearch_2D import plotting, env


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

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
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

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        #以f值为堆索引 堆（索引，坐标）
        heapq.heappush(self.OPEN,(self.f_value(self.s_start), self.s_start))
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN) #弹出索引最小值
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break
            
            for s_n in self.get_neighbor(s):
                #若s至s_n发生碰撞，则cost=inf
                new_cost = self.g[s] + self.cost(s, s_n)

                #若s_n 不在OPEN列表中，则其没有g值，将其g设为inf，便于下一步将该点加入OPEN中
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED
    

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]


    def f_value(self,s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        return self.g[s]+self.weight*self.heuristic(s)
    
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

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
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

    


    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        self.path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            self.path.append(s)

            if s == self.s_start:
                break
        # print("path length:",np.array(self.path).size/2)
        # print(np.array(self.path)[1][1])
        self.path=self.delete_colilinear_points(self.path)
        # print("path length:",np.array(self.path))

        self.path=self.delete_redundant_inflection_points(self.path)
        return list(self.path)

    def delete_colilinear_points(self, path):
       
        path_after_delete_colilinear_points=[]
        # path_length路径长度，不同于list[list],该路径是list[tuple()]
        # 不能使用(np.array(path).size)[0]获取
        path_length=int((np.array(path).size)/2)
        for i in range(0,path_length):            
            if(i==0):                
                path_after_delete_colilinear_points.append(path[i])
            elif(i==path_length-1):                
                path_after_delete_colilinear_points.append(path[i])
            elif(i>=1 and i<=path_length-2):                
                if((path[i+1][0]-path[i][0])!=(path[i][0]-path[i-1][0])) or ((path[i+1][1]-path[i][1])!=(path[i][1]-path[i-1][1])):
                    path_after_delete_colilinear_points.append(path[i])
        return path_after_delete_colilinear_points
    
    def delete_redundant_inflection_points(self, path):
        path_after_delete_redundant_inflection_points=[]
        
        path_length=int(np.array(path).size/2)
        path_after_delete_redundant_inflection_points.append(path[path_length-1])
        do_delete=1
        n=path_length
        # print(n)
        while(do_delete):
            # print(n)
            for i in range(0,n):
                # time.sleep(1)
                
                if(self.is_line_collision(path[i],path[n-1])==True):
                    continue
                else:
                    path_after_delete_redundant_inflection_points.append(path[i])
                    if(i==0):
                        # print("i==0")
                        do_delete=0
                    n=i+1                    
                    break               

        return path_after_delete_redundant_inflection_points

    
def main():
    s_start = (5, 25)
    s_goal = (45, 25)
    # s_start = (2, 2)
    # s_goal = (38, 2)
    

    astar = AStar(s_start, s_goal, "euclidean")
    # print("is line collision? ",astar.is_line_collision(s_start,s_goal))

    plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    
    main()