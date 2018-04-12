#!coding=utf-8

import math
import numpy as np
import sys
from ClassMap import ClassMap


def rtt(map_obj, k, delta_q):
    G = [map_obj.begin]
    edge = []
    path = []
    for i in range(k):
        # RAND_CONF()
        qrand = map_obj.random_gen_point()
        # NEAREST_VERTEX(qrand, G)
        # import pdb;pdb.set_trace()
        qnear = get_qnear(qrand, G)
        if False not in (qnear == qrand):
            continue
        # NEW_CONF(qnear, qrand, Δq)
        qnew = get_qnew(qnear, qrand, delta_q)
        move_vet = qnew - qnear

        if path_in_free_space(map_obj, qnear, move_vet, delta_q):
            G.append(qnew)
            edge.append((qnew, qnear))
            
            if qnew is map_obj.goal or is_glod_near_new_edge(qnear, move_vet, delta_q, map_obj.goal):
                if qnew is not map_obj.goal:
                    G = G[:-2] + [map_obj.goal]
                # import pdb;pdb.set_trace()
                path = connect_path(edge, G)
                return G, edge, path

    print("cannot find the solution@")
    return G, edge, path

def get_qnear(qrand, G):
    min_dist = sys.maxsize
    qnear = G[0]
    for i in range(len(G)):
        dist = np.linalg.norm(G[i] - qrand)
        if min_dist > dist:
            min_dist = dist
            qnear = G[i]

    return qnear

def get_qnew(qnear, qrand, delta_q):
    # import pdb;pdb.set_trace()
    move_vet = qrand - qnear
    # print(move_vet[0])
    # import pdb;pdb.set_trace()
    # if move_vet[0] == 0 and move_vet[1] == 0:
    #     import pdb;pdb.set_trace()
    ve = move_vet / np.linalg.norm(move_vet)
    qnew = qnear + ve*delta_q
    return qnew

def path_in_free_space(map_obj, qnear, move_vet, step_dist):

    mp = 10
    # v = qnew - qnear
    # step_dist = np.linalg.norm(v)
    ve = move_vet / step_dist
    delta = step_dist / mp

    cur_point = qnear
    # import pdb;pdb.set_trace()
    # print(cur_point)
    for i in range(mp):
        cur_point = cur_point + (delta * ve)
        # print(ve)
        # print(cur_point)
        # print(int(cur_point[0]))
        # print(move_vet)
        if not map_obj.in_free_space(int(cur_point[1]), int(cur_point[0])):
            return False

    return True

def is_glod_near_new_edge(qnear, move_vet, step_dist, qgoal):
    ve = move_vet / step_dist
    dist_2goal = np.linalg.norm(qgoal - qnear)
    if dist_2goal > step_dist:
        return False
    else:
        # last_vet = qnear + (dist_2goal*ve)
        # if last_vet == move_vet:
        #     return
        return True

def connect_path(edge, G):
    path = []
    count = 0
    # edge = [(qnew, qnear)]
    # import pdb;pdb.set_trace()
    cur_point = edge[-1][0]
    pre_point = edge[-1][1]
    edge_count = len(edge)
    path.append(cur_point)
    while pre_point is not edge[0][1]:
        if count > edge_count:
            print("cannot find out the solution")
            break
        for item in edge:
            if item[0] is pre_point:
                pre_point = item[1]
                path.append(pre_point)
                break

        count += 1
    return path

map_obj = ClassMap(10, 10)
map_obj.generate_obstacle()
map_obj.init_goal_and_begin_point()

k = 100
delta_q = 1

G, e, p = rtt(map_obj, k, delta_q)
print(p)
map_obj.draw_map(path=p)
