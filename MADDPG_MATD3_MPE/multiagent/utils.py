"""
1，本文件是uav_conf_env的辅助文件
2，本文件在environment.py文件中被调用
"""

import random
import numpy as np
import math
import copy

'''
1，generate_random_coordinates函数用于生成随机的坐标[x, y]
2，min_val_x, max_val_x, min_val_y, max_val_y分别是x、y坐标的下上限

返回x、y坐标在输入范围内的坐标
'''

def generate_random_coordinates(min_val_x, max_val_x, min_val_y, max_val_y):
    x = random.uniform(min_val_x, max_val_x)
    y = random.uniform(min_val_y, max_val_y)
    return [x, y]


def generate_random_val(min_val, max_val):
    return random.uniform(min_val, max_val)


'''
1，formation_construct函数用于blue编队保持
2，delta_x, delta_y 每个agent相对于中心leader的x、y距离

返回处理了blue的完整action  （red+blue）
'''

def formation_construct(num_good, num_adv, action_set, delta_x, delta_y, max_speed, gamma, d_max, old_new_obs_n):
    # get action
    action_n = copy.deepcopy(action_set)  ### 这个变量按顺序存储所有agent的速度（red1,red2,...,blue1,blue2,...）
    for cnt_i in range(num_good - 1):
        action_n[-(cnt_i + 2)] = np.zeros(5)
    # Adjenct need to fix!
    A = [[0] * (num_good - 1) + [1]] * (num_good - 1) + [[0] * num_good]  # Adjenct Matrix

    v_temp = np.zeros([num_good, 2])  ### 暂存速度运算中间量
    # leader
    # action_n[-1] = action_leader
    # action_n[num_adv + num_good - 1]=[0,0,.3,0,.3]    #[unknown, x-right,x-left,y-up,y-down]
    # follower
    for i in range(num_good - 1):
        sum_delta_x, sum_delta_y, sum_edge_weight = .0, .0, .0
        for j in range(num_good):
            if A[i][j] == 1:
                # print(old_new_obs_n[num_adv+j][4+num_adv*2+i*2+1-1])
                delta_pos_ij_x = old_new_obs_n[num_adv + j][2 + num_adv * 2 + i * 2 + 1 - 1] if i < j \
                    else old_new_obs_n[num_adv + j][2 + num_adv * 2 + i * 2 + 1 - 1 - 2] if i > j \
                    else 0
                delta_pos_ij_y = old_new_obs_n[num_adv + j][2 + num_adv * 2 + i * 2 + 2 - 1] if i < j \
                    else old_new_obs_n[num_adv + j][2 + num_adv * 2 + i * 2 + 2 - 1 - 2] if i > j \
                    else 0

                w_ij = 2 - np.exp(-(np.square(-delta_pos_ij_x - (delta_x[j] - delta_x[i])) +
                                    np.square(
                                        -delta_pos_ij_y - (delta_y[j] - delta_y[i]))))  # edge weighted calculation
                sum_delta_x += A[i][j] * w_ij * (-delta_pos_ij_x - (delta_x[j] - delta_x[i]))
                sum_delta_y += A[i][j] * w_ij * (-delta_pos_ij_y - (delta_y[j] - delta_y[i]))
                # sum_edge_weight+=w_ij
                # print(w_ij)
        dist_fol = np.sqrt(np.square(sum_delta_x) + np.square(sum_delta_y))
        if dist_fol > d_max:
            dist_fol = d_max
        th = math.atan2(sum_delta_y, sum_delta_x)  # ouput is rad
        v_temp[i][0] = action_n[-1][1] + gamma * dist_fol * np.cos(th)  ### 注意这里相对于leader的速度
        # constraint  ### 单独限制x、y方向速度不合理，应限制速度模值，但是限制模值怎么具体回到x和y？不限制似乎也可行，因为编队与leader一致
        # if v_temp[i][0] > max_speed:
        #     v_temp[i][0] = max_speed
        # if v_temp[i][0] < -max_speed:
        #     v_temp[i][0] = -max_speed
        action_n[num_adv + i][1] = v_temp[i][0]
        v_temp[i][1] = action_n[-1][3] + gamma * dist_fol * np.sin(th)
        # if v_temp[i][1] > max_speed:
        #     v_temp[i][1] = max_speed
        # if v_temp[i][1] < -max_speed:
        #     v_temp[i][1] = -max_speed
        action_n[num_adv + i][3] = v_temp[i][1]
        action_n[num_adv + i][0], action_n[num_adv + i][2], action_n[num_adv + i][4] = 0, 0, 0

    return action_n


def four_dir_generate_random_coordinates(screen_size, num_rule, dist_rule, num_adv, num_food):
    '''
    在四个象限随机生成红蓝双方和目标位置
    输入坐标范围screen_size 蓝方数量num_rule 蓝方follower和leader的距离dist_rule 红方数量num_adv 目标点数量num_food
    红方和目标在同一个象限，蓝方在对角线
    返回坐标(np.array格式) blue_swarm_pos, red_swarm_pos, food_check_pos
    '''
    x = random.uniform(screen_size[0], screen_size[1])
    y = random.uniform(screen_size[0], screen_size[1])
    x_mid = sum(screen_size) / 2
    y_mid = sum(screen_size) / 2
    ##########################
    #    2     #      1      #
    #          #             #
    ##########################
    #    3     #      4      #
    #          #             #
    ##########################
    if x < x_mid and y > y_mid:  # 左上
        blue_mark = 2
        red_mark = 4
    elif x > x_mid and y > y_mid:  # 右上
        blue_mark = 1
        red_mark = 3
    elif x > x_mid and y < y_mid:  # 右下
        blue_mark = 4
        red_mark = 2
    elif x < x_mid and y < y_mid:  # 左下
        blue_mark = 3
        red_mark = 1

    blue_pos = [x, y]
    blue_swarm_pos = []
    blue_swarm_pos.append(blue_pos)
    # print("num_blue:", num_rule)
    for i in range(num_rule-1):
        angle = 2 * np.pi * i / (num_rule-1)
        x_blue = x + dist_rule * np.cos(angle)
        y_blue = y + dist_rule * np.sin(angle)
        blue_swarm_pos.append([x_blue, y_blue])

    red_swarm_pos = []
    food_check_pos = []
    if red_mark == 2:  
        for _ in range(num_adv):
            # # red在地图上左上半部分，方程 y>x
            # x_red = generate_random_val(screen_size[0], screen_size[1])
            # y_red = generate_random_val(x_red, screen_size[1])  # 1>y>x>-1
            # red在第二象限
            x_red = generate_random_val(screen_size[0], x_mid)
            y_red = generate_random_val(y_mid, screen_size[1])
            red_swarm_pos.append([x_red, y_red])
        for _ in range(num_food):
            check_pos = generate_random_coordinates(screen_size[0], x_mid, y_mid, screen_size[1]) # 左上
            food_check_pos.append(check_pos)
    elif red_mark == 1:  
        for _ in range(num_adv):
            # # red在地图上右上半部分，方程 y>-x
            # x_red = generate_random_val(screen_size[0], screen_size[1])
            # y_red = generate_random_val(-x_red, screen_size[1])  # 1>y>-x>-1
            # red在第一象限
            x_red = generate_random_val(x_mid, screen_size[1])
            y_red = generate_random_val(y_mid, screen_size[1])
            red_swarm_pos.append([x_red, y_red])
        for _ in range(num_food):
            check_pos = generate_random_coordinates(x_mid, screen_size[1], y_mid, screen_size[1]) #右上
            food_check_pos.append(check_pos)
    elif red_mark == 4:  
        for _ in range(num_adv):
            # # red在地图上右下半部分，方程 y<x
            # x_red = generate_random_val(screen_size[0], screen_size[1])
            # y_red = generate_random_val(screen_size[0], x_red)  # -1<y<x<1
            # red在第四象限
            x_red = generate_random_val(x_mid, screen_size[1])
            y_red = generate_random_val(screen_size[0], y_mid)
            red_swarm_pos.append([x_red, y_red])
        for _ in range(num_food):
            check_pos = generate_random_coordinates(x_mid, screen_size[1], screen_size[0], y_mid)
            food_check_pos.append(check_pos)
    elif red_mark == 3:  
        for _ in range(num_adv):
            # # red在地图上左下半部分，方程 y<-x
            # x_red = generate_random_val(screen_size[0], screen_size[1])
            # y_red = generate_random_val(screen_size[0], -x_red)  # -1<y<-x<1
            # red在第三象限
            x_red = generate_random_val(screen_size[0], x_mid)
            y_red = generate_random_val(screen_size[0], y_mid)
            red_swarm_pos.append([x_red, y_red])
        for _ in range(num_food):
            check_pos = generate_random_coordinates(screen_size[0], x_mid, screen_size[0], y_mid)
            food_check_pos.append(check_pos)

    return np.array(blue_swarm_pos), np.array(red_swarm_pos), np.array(food_check_pos)