
#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-12-14
# Modified    :   2017-12-14
# Version     :   1.0

import csv
import numpy as np
import pickle
import os


class DataHolder(object):
    TICK = 'tick'
    COORDINATE = 'coordinate'
    SPEED = 'speed'
    TYPE = 'type'

    def __init__(self, fname, holder_name):
        self.holder_name = holder_name
        self.file = fname
        self.car_traj = {}
        self.time_offset = 0
        self.precision = 10**0

    @staticmethod
    def create_instance(cls, fname='data/trajectory.csv', holder_name='./data_pkl/data_holder.pkl'):
        if os.path.lexist(holder_name):
            with open(holder_name, 'rb') as f:
                holder = pickle.load(f)
        else:
            holder = DataHolder(fname, holder_name)
            holder.load_data()
        return holder

    def coord_formater(self, coord):
        return int(float(item[2]) * self.precision)

    def load_data(self):
        """
        format:
            vehicle-id,time,x-coordinate,y-coordinate,speed,category
            4c0c4745067197be22182d262b44f48a,1493859164,521696.473915,55061.506951,6.6,0,
        """
        r = csv.reader(open(self.file, 'r'))
        line = next(r)
        for item in r:
            if item[0] in self.car_traj:
                self.car_traj[item[0]][self.TICK].append(int(item[1]) - self.time_offset)
                self.car_traj[item[0]][self.COORDINATE].append([self.coord_formater(item[2]), self.coord_formater(item[3])])
                self.car_traj[item[0]][self.SPEED].append([float(item[4])])
                self.car_traj[item[0]][self.TYPE].append(int(item[5]))
            else:
                self.car_traj[item[0]] = {}
                self.car_traj[item[0]][self.TICK] = (int(item[1]) - self.time_offset)
                self.car_traj[item[0]][self.COORDINATE] = ([self.coord_formater(item[2]), self.coord_formater(item[3])])
                self.car_traj[item[0]][self.SPEED] = ([float(item[4])])
                self.car_traj[item[0]][self.TYPE] = (int(item[5]))
        with open(self.holder_name, 'wb') as f:
            pickle.dump(self, f)
