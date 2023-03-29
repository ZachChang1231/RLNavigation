# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : custom_envs.py
# Time       ：2023/3/16 下午5:49
# Author     ：Zach
# version    ：python 3.8
# Description：
"""

import copy
import os
import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gym import spaces
from shapely import MultiLineString, LineString
from shapely import geometry as geo
from shapely import STRtree

from envs.multiprocessing_env import SubprocVecEnv
from model.utils import print_line


class MultiCusEnv(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        _env = [self._make_env() for _ in range(cfg.num_processes)]
        _env = SubprocVecEnv(_env)

        self._env = _env
        self.observation_space, self.action_space = _env.observation_space, _env.action_space

        self.seed()

    def seed(self):
        self._env.seed([self.cfg.seed + i for i in range(self.cfg.num_processes)])

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def close(self):
        self._env.close()

    def _make_env(self):
        def _thunk():
            env = CusEnv(self.cfg, self.logger)
            return env

        return _thunk


class CusEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'array'],
        'video.frames_per_second': 2
    }

    def __init__(self, cfg, logger, **kwargs):
        self.rnd = np.random.RandomState(cfg.seed)
        self.cfg = cfg
        self.logger = logger

        self.env_list = self._get_env_list()

        if cfg.map_type == 'from_image':
            assert cfg.env_name in self.env_list, "Undefined environment!"
        else:
            pass  # TODO load map from manual functions

        self.action_space = spaces.Discrete(cfg.turning_angle_num + 1)
        self.action_info = {
            "turning": np.linspace(-cfg.turning_range, cfg.turning_range, cfg.turning_angle_num),
            "forward": 5
        }

        self.observation_space = spaces.Box(np.array([0 for _ in range(cfg.laser_num * 2 + 2)], dtype=np.float32),
                                            np.array([1 for _ in range(cfg.laser_num * 2 + 2)], dtype=np.float32))

        self.init_state_memory = None
        self.kwargs_memory = dict()
        self.reset_num_count = 0
        self.step_num = 0

        self.state = {"pre_laser_state": [], "laser_state": [], "laser_info": [], "target_state": [], "self_state": []}
        """
        state = {
            "pre_laser_state": [s0, s1, ..., sn]  (s in [0, 1], ds in [-1, 1]),
            "laser_state": [s0, s1, ..., sn]  (s in [0, 1], ds in [-1, 1]),
            "target_state": [distance, theta]  (distance in [0, 1], theta in [-1, 1]),
            "self_state": [vx, vy, px, py],
            "laser_info": [[end_x, end_y] * laser_num]
        }
        """
        self.temp_state = {"dis2goal": 0, "dis2coll": 0}

        self.map_info = {
            "loaded": False,
            "shape": np.array(cfg.env_size.split('*')).astype(float),
            "coll_num": 0,
            "node_group": [],
            "coll_group": [],
            "coll_tree": STRtree([]),
        }
        """
        map_info = {
            "shape": [x, y],
            "coll_num": int,
            "node_group": [np.ndarray shape(n0, 2), np.ndarray shape(n1, 2), ...],
            "coll_group": [Polygon, ...],
            "coll_tree": STRtree,
        }
        """
        self.target_position = np.array([1, 1])
        self.position = np.array([1, 1])
        self.velocity = np.array([1, 0])

        self.figure, self.axes = None, None

        show = kwargs["show"] if "show" in kwargs.keys() else False
        self.load_map(show)

    def step(self, action):
        assert self.action_space.contains(action), "Illegal action!"
        assert self.state["laser_state"], "Environment should reset before step!"

        if action == self.action_space.n - 1:
            self._forward()
        else:
            self._turn(action)

        laser_array = self._get_laser_posture()
        laser_state, laser_info = self._get_laser_state(laser_array)
        distance, theta = self._get_target_info(trans=True)

        self.state["pre_laser_state"] = copy.deepcopy(self.state["laser_state"])
        self.state["laser_state"] = laser_state
        self.state["laser_info"] = laser_info
        self.state["target_state"] = [distance, theta]
        self.state["self_state"] = [self.velocity[0], self.velocity[1], self.position[0], self.position[1]]

        reward = self._get_reward()

        terminated = 1 if (self._collision_detection(self._self_node) or self._arrive) else 0
        self.step_num += 1

        if self.step_num > self.cfg.max_steps:
            truncated = 1
        else:
            truncated = 0

        return self._state_integration(), reward, terminated, truncated, {}

    def reset(self, **kwargs):
        assert self.map_info["loaded"], "Map not loaded!"
        if (self.reset_num_count == 0) or \
                (self.cfg.initial_interval and self.reset_num_count % self.cfg.initial_interval == 0):

            assert_dict = {"position": 0, "target_position": 0, "velocity": 0}

            if kwargs and not self.kwargs_memory:
                self.kwargs_memory = kwargs
            if not kwargs and self.kwargs_memory:
                kwargs = self.kwargs_memory
            if kwargs:
                assert False not in [True if i in list(assert_dict.keys()) else False for i in list(kwargs.keys())], \
                    "Unsupported key in kwargs!"
                for key in assert_dict.keys():
                    if key in kwargs.keys():
                        assert_dict[key] = 1
                        assert isinstance(kwargs[key], str), "The input kwargs should be string!"
                        if ":" not in kwargs[key]:
                            try:
                                value = np.array(kwargs[key].split(',')).astype(float)
                            except Exception as e:
                                # print(e)
                                raise ValueError("Cannot resolve the input value!")
                            if "position" in key:
                                assert self._check_point_available(value), '{} not available!'.format(key.title())
                            else:
                                value = value / np.sqrt(np.sum(np.square(value)))
                            exec("self." + key + "= value")
                        else:
                            if kwargs[key].count(':') > 1:
                                assert_dict[key] = 0
                                continue
                            assert key != 'velocity', "Please clearly define the velocity!"
                            x, y = kwargs[key].split(',')
                            if ":" in x:
                                point = self._get_available_points(y=float(y))
                            else:
                                point = self._get_available_points(x=float(x))
                            exec("self." + key + "= point")

            for key, value in assert_dict.items():
                if not value:
                    if "position" in key:
                        exec("self." + key + "= self._get_available_points()")
                    else:
                        random_theta = self.rnd.random() * np.pi * 2
                        self.velocity = np.array([np.cos(random_theta), np.sin(random_theta)])

            laser_array = self._get_laser_posture()
            laser_state, laser_info = self._get_laser_state(laser_array)

            distance, theta = self._get_target_info(trans=True)

            self.state["pre_laser_state"] = [0 for _ in range(self.cfg.laser_num)]
            self.state["laser_state"] = laser_state
            self.state["laser_info"] = laser_info
            self.state["target_state"] = [distance, theta]
            self.state["self_state"] = [self.velocity[0], self.velocity[1], self.position[0], self.position[1]]

            self.temp_state["dis2goal"] = self.state["target_state"][0]
            self.temp_state["dis2coll"] = self._calculate_min_distance2coll(self._self_node)

            self.init_state_memory = copy.deepcopy(
                [self.position, self.target_position, self.velocity, self.state, self.temp_state])
        else:
            assert self.init_state_memory is not None, "Memory not stored!"
            self.position, self.target_position, self.velocity, self.state, self.temp_state = copy.deepcopy(
                self.init_state_memory)
        self.reset_num_count += 1
        self.step_num = 0

        return self._state_integration()

    def render(self, mode='human'):
        if mode == 'human':
            if self.figure is None:
                plt.ion()
                self.figure, self.axes = plt.subplots()
            plt.cla()
            self.axes.set_xlim(0, self.map_info["shape"][0])
            self.axes.set_ylim(0, self.map_info["shape"][1])
            self.axes.set_aspect(1)

            for i in range(self.map_info["coll_num"]):
                group_edge_node = self.map_info["node_group"][i]
                self.axes.fill(group_edge_node[:, 0], group_edge_node[:, 1], 'k')

            robot = plt.Circle((self.position[0], self.position[1]), self.cfg.robot_size, color='r')
            initial_position = plt.Circle((self.init_state_memory[0][0], self.init_state_memory[0][1]), self.cfg.robot_size+1, color='y')
            target_position = plt.Circle((self.init_state_memory[1][0], self.init_state_memory[1][1]), self.cfg.robot_size+1, color='g')
            self.axes.add_artist(robot)
            self.axes.add_artist(initial_position)
            self.axes.add_artist(target_position)

            for laser in self.state["laser_info"]:
                self.axes.plot([self.position[0], laser[0]], [self.position[1], laser[1]], color='b')

            plt.pause(0.01)
            # plt.show()

        elif mode == 'array':
            pass
            # TODO return rgb array
        else:
            raise NotImplementedError

    def save_fig(self, index, save_path):
        matplotlib.use("Agg")
        figure, axes = plt.subplots()
        plt.cla()
        axes.set_xlim(0, self.map_info["shape"][0])
        axes.set_ylim(0, self.map_info["shape"][1])
        axes.set_aspect(1)

        for i in range(self.map_info["coll_num"]):
            group_edge_node = self.map_info["node_group"][i]
            axes.fill(group_edge_node[:, 0], group_edge_node[:, 1], 'k')

        robot = plt.Circle((self.position[0], self.position[1]), self.cfg.robot_size, color='r')
        initial_position = plt.Circle((self.init_state_memory[0][0], self.init_state_memory[0][1]),
                                      self.cfg.robot_size + 1, color='y')
        target_position = plt.Circle((self.init_state_memory[1][0], self.init_state_memory[1][1]),
                                     self.cfg.robot_size + 1, color='g')
        axes.add_artist(robot)
        axes.add_artist(initial_position)
        axes.add_artist(target_position)

        for laser in self.state["laser_info"]:
            axes.plot([self.position[0], laser[0]], [self.position[1], laser[1]], color='b')

        plt.savefig(os.path.join(save_path, "{}.png".format(index)))
        plt.close()
        # plt.show()
        # plt.close()

    def close(self):
        plt.ioff()

    def seed(self, seed):
        self.rnd = np.random.RandomState(seed)

    def _forward(self):
        self.position = self.position + self.action_info["forward"] * self.velocity

    def _turn(self, index):
        turning_angle = self.action_info["turning"][index]
        self.velocity = self._rotate(self.velocity, turning_angle)

    @property
    def _arrive(self):
        if self.temp_state["dis2goal"] <= (self.cfg.robot_size * 2):
            return True
        else:
            return False

    def _get_reward(self):
        """
        1. Positive rewards for arriving at goals and movement close to these goals
        2. Negative rewards for collision penalty or movement too close to an obstacle
        3. Negative reward for time step penalty (encourage a faster movement)
        """
        curr_dis2goal = self.state["target_state"][0]
        curr_dis2coll = self._calculate_min_distance2coll(self._self_node)
        pre_dis2goal = self.temp_state["dis2goal"]
        pre_dis2coll = self.temp_state["dis2coll"]
        self.temp_state["dis2goal"] = curr_dis2goal
        self.temp_state["dis2coll"] = curr_dis2coll

        if self._arrive:
            arrive_reward = 100
        else:
            arrive_reward = (pre_dis2goal - curr_dis2goal) * self.cfg.arrive_reward_weight

        if curr_dis2coll == 0:
            collision_reward = -100
        else:
            # collision_reward = -(pre_dis2coll - curr_dis2coll) * self.cfg.collision_reward_weight
            collision_reward = 0

        time_step_reward = -self.cfg.time_step_reward_weight

        return arrive_reward + collision_reward + time_step_reward

    def _state_integration(self, normalize=True):
        distance, theta = self.state["target_state"]
        if normalize:
            theta = (theta + np.pi) / np.pi  # [0, 1]
            distance = distance / self._get_distance([0, 0], self.map_info["shape"])  # [0, 1]
        state = self.state["pre_laser_state"] + self.state["laser_state"] + [distance, theta]
        return np.array(state)

    def _get_env_list(self):
        env_list = []
        map_list = os.listdir(self.cfg.map_path)
        for map_ in map_list:
            map_name, format_ = map_.split('.')
            if format_ == 'jpg':
                env_list.append(map_name)
        return env_list

    def load_map(self, show=False):
        image = Image.open(os.path.join(self.cfg.map_path, '{}.jpg'.format(self.cfg.env_name)))
        image = image.convert('L')
        map_ = np.array(image)
        self._clean(map_)

        if self.cfg.shape_fixed:
            self.map_info["shape"] = np.array(self.cfg.env_size.split('*')).astype(float)
        else:
            self.map_info["shape"] = np.array(map_.shape[::-1])

        print_line(self.logger, 'load')
        self._segment(map_, show)

        print_line(self.logger, 'line')
        self.map_info["loaded"] = True

    @staticmethod
    def _clean(map_):
        threshold = 255 / 2
        map_[map_ <= threshold] = 1
        map_[map_ > threshold] = 0

    def _segment(self, map_, show=False):
        contours, _ = cv2.findContours(map_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        coll_num = len(contours)
        self.map_info["coll_num"] = coll_num

        coll_list = []
        node_list = []
        for i in range(coll_num):
            epsilon = 0.01 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            group_edge_node = approx.squeeze()
            group_edge_node[:, 1] += (group_edge_node[:, 1] == map_.shape[0] - 1).astype(int)

            if self.cfg.shape_fixed:
                group_edge_node = group_edge_node * self.map_info["shape"] / np.array(map_.shape)[::-1]
            group_edge_node[:, 1] = self.map_info["shape"][1] - group_edge_node[:, 1]
            node_list.append(group_edge_node)

            poly = []
            for j in range(group_edge_node.shape[0]):
                poly.append(tuple(group_edge_node[j]))
            poly.append(tuple(group_edge_node[0]))
            coll_list.append(geo.Polygon(poly))
        edge = [self.map_info["shape"][0], self.map_info["shape"][1]]
        coll_list.append(
            geo.Polygon(shell=[(-1, -1), (edge[0] + 1, -1), (edge[0] + 1, edge[1] + 1), (-1, edge[1] + 1), (-1, -1)],
                        holes=[[(0, 0), (edge[0], 0), (edge[0], edge[1]), (0, edge[1]), (0, 0)]])
        )
        self.map_info["node_group"] = node_list
        self.map_info["coll_group"] = coll_list
        self.map_info["coll_tree"] = STRtree(coll_list)

        self._check_map_available()

        if show:
            self._show_map()

    def _check_map_available(self):
        min_dis = 1e6
        for i in range(-1, self.map_info["coll_num"]):
            for j in range(i + 1, self.map_info["coll_num"]):
                dis = self.map_info["coll_group"][i].distance(self.map_info["coll_group"][j])
                if dis < min_dis and dis != 0:
                    min_dis = dis
        single_coll_flag = 1 if self.map_info["coll_num"] == 1 and min_dis == 1e6 else 0
        self.map_info["min_dis"] = min_dis

        available = True if self.cfg.robot_size * 2 <= min_dis else False
        assert available, 'Do not have available path for this image! The narrowest distance is {:.2f} ' \
                          'while the robot diameter is {:.2f}.'.format(min_dis, self.cfg.robot_size * 2)
        # TODO if single coll flag, a new way should be built to calculate min distance
        self.logger.info(
            "Map is available for navigation, the narrowest distance is {:.2f}.".format(self.map_info["min_dis"]))
        self.logger.info("Env name: {}, Env shape: {}, Collision num: {}".format(self.cfg.env_name,
                                                                                 self.map_info["shape"],
                                                                                 self.map_info["coll_num"]))

    def _show_map(self):
        colors = np.linspace(0, 0.8, self.map_info["coll_num"])

        figure, axes = plt.subplots()
        axes.set_xlim(0, self.map_info["shape"][0])
        axes.set_ylim(0, self.map_info["shape"][1])
        axes.set_aspect(1)
        for i, v in enumerate(colors):
            group_edge_node = self.map_info["node_group"][i]
            axes.fill(group_edge_node[:, 0], group_edge_node[:, 1], str(v))
        # plt.savefig('./1.png')
        plt.show()

    def _calculate_min_distance2coll(self, point):
        index, distance = self.map_info["coll_tree"].query_nearest(point, return_distance=True)
        return distance[0]

    def _collision_detection(self, point, return_coll=False):
        index = self.map_info["coll_tree"].query(point, predicate="intersects")
        is_coll = True if len(index) > 0 else False
        coll = list(self.map_info["coll_tree"].geometries.take(index))
        if return_coll:
            return is_coll, coll
        else:
            return is_coll

    @property
    def _self_node(self):
        return geo.Point(self.position).buffer(self.cfg.robot_size)

    def _get_laser_posture(self):
        laser_array = []
        theta = np.linspace(-self.cfg.laser_range, self.cfg.laser_range, self.cfg.laser_num)
        for i in range(self.cfg.laser_num):
            laser_direction = self._rotate(self.velocity, theta[i])
            laser_end = self.position + self.cfg.laser_length * laser_direction
            laser_array.append(geo.LineString([self.position, laser_end]))
        return laser_array

    def _get_laser_state(self, laser_list):
        laser_state = []
        laser_info = []
        if self._collision_detection(geo.Point(self.position)):
            laser_state = [0] * self.cfg.laser_num
            laser_info = [[self.position[0], self.position[1]]] * self.cfg.laser_num
        else:
            for laser in laser_list:
                is_coll, coll = self._collision_detection(laser, return_coll=True)
                if not is_coll:
                    laser_state.append(1)
                    laser_info.append(list(laser.coords[1]))
                    continue
                dis = 0
                differ_laser = None
                differ = laser.difference(geo.MultiPolygon(coll))
                if isinstance(differ, LineString):
                    dis = differ.length
                    differ_laser = differ
                elif isinstance(differ, MultiLineString):
                    for line in list(differ.geoms):
                        if self._self_node.distance(line) == 0:
                            dis = line.length
                            differ_laser = line
                            break
                else:
                    raise ValueError("Shapely object not defined!")

                laser_state.append(dis / self.cfg.laser_length)
                laser_info.append(list(differ_laser.coords[1]))

        return [1 - length for length in laser_state], laser_info

    def _get_available_points(self, num=1, **kwargs):
        point_list = []
        for _ in range(num):
            flag = 0
            count = 0
            while count < 1e4:
                if not kwargs:
                    point = self.rnd.random(2) * self.map_info["shape"]
                elif 'x' in kwargs.keys():
                    point = np.array([kwargs['x'], self.rnd.random() * self.map_info["shape"][1]])
                elif 'y' in kwargs.keys():
                    point = np.array([self.rnd.random() * self.map_info["shape"][0], kwargs['y']])
                else:
                    raise ValueError("Key undefined!")
                count += 1
                if self._check_point_available(point):
                    flag = 1
                    break
            assert flag, "Do not find an available point!"
            point_list.append(point)
        if num == 1:
            return point_list[0]
        else:
            return point_list

    def _check_point_available(self, point):
        if isinstance(point, np.ndarray):
            point = geo.Point(point).buffer(self.cfg.robot_size)
        assert isinstance(point, geo.base.BaseGeometry), "Point type unsupported!"
        return not self._collision_detection(point)

    def _get_target_info(self, trans=True):
        target_vec = self.target_position - self.position
        if trans:
            theta = self._get_vector_theta(vec=self.velocity, target_vec=target_vec)
        else:
            theta = self._get_vector_theta(vec=target_vec)
        distance = self._get_distance(target_vec)
        return distance, theta

    # @nb.jit  TODO use nb.jit to accelerate
    def _get_vector_theta(self, vec, target_vec=np.array([1, 0])):
        absolute_target_vec = self._get_distance(target_vec)
        absolute_vec = self._get_distance(vec)
        if absolute_target_vec != 0:
            if (target_vec == vec).all():
                return 0
            else:
                cos_theta = (target_vec[0] * vec[0] + target_vec[1] * vec[1]) / absolute_target_vec / absolute_vec
                if 1 < cos_theta < 1 + 1e-5:
                    cos_theta = 1
                elif -1 - 1e-5 < cos_theta < -1:
                    cos_theta = -1
                assert -1 <= cos_theta <= 1
                theta = np.arccos(cos_theta)
                direction = vec[0] * target_vec[1] - vec[1] * target_vec[0]
                if direction > 0:
                    return -theta  # -pi ~ pi
                else:
                    return theta
        else:
            raise ValueError("Illegal velocity!")

    def add_moving_coll(self):
        pass

    @staticmethod
    def _get_distance(vec1, vec2=None):
        if vec2 is not None:
            distance = np.sqrt(np.sum(np.square(np.array(vec1) - np.array(vec2))))
        else:
            distance = np.sqrt(np.sum(np.square(vec1)))
        return distance

    @staticmethod
    def _rotate(vector, theta):
        return vector.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
