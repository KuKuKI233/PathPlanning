"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = 101  # size of background
        self.y_range = 101
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))
        
        for i in range(0,25):
            obs.add((25, i))
            obs.add((24, i))
        
        for i in range(60,81):
            obs.add((25, i))
            obs.add((24, i))

        for i in range(50,75):
            obs.add((50, i))
            obs.add((51, i))
        for i in range(40,60):
            obs.add((i, 50))
            obs.add((i, 51))

        for i in range(60,87):
            obs.add((i, 30))
            obs.add((i, 31))
        for i in range(15,30):
            obs.add((85, i))
            obs.add((86, i))

        for i in range(70,80):
            obs.add((70, i))
            obs.add((71, i))
        for i in range(70,90):
            obs.add((i, 70))
            obs.add((i, 71))
        for i in range(50,80):
            obs.add((90, i))
            obs.add((91, i))

        # for i in range(10, 21):
        #     obs.add((i, 15))
        # for i in range(15):
        #     obs.add((20, i))

        # for i in range(15, 30):
        #     obs.add((30, i))
        # for i in range(16):
        #     obs.add((40, i))

        return obs
