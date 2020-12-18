import numpy as np
import math


def generate_board(obstacle_list: list, start: tuple, end: tuple):
    # '0' is empty space
    # '1' is the player
    # '2' is the end goal
    # '3' is an obstacle
    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]

    obstacle_list = np.argwhere(maze == 1)
    if start in obstacle_list or end in obstacle_list:
        return None
    board = np.zeros((24, 24), dtype=int)
    board[start] = 1
    board[end] = 2
    for obstacle in obstacle_list:
        board[tuple(obstacle)] = 3

    return board


def find_adjacent(board, curr_loc, visited=[]):
    # returns all adjacent tiles that are not obstacles
    deltas = [[0, 1], (0, -1), (1, 0), (-1, 0)]
    adj_list = []
    for delta in deltas:
        new_loc = np.array(curr_loc) + np.array(delta)
        correct_tile = np.all(new_loc >= 0) and np.all(new_loc < 24) and board[tuple(new_loc)] in [0, 2]
        if correct_tile and not (tuple(new_loc) in visited):
            adj_list.append(tuple(new_loc))
    return adj_list


def BFS(board):
    start = np.where(board == 1)
    curr_loc = (start[0][0], start[1][0])

    visited = []
    queue = []
    queue.append([curr_loc])
    while queue:
        path = queue.pop(0)
        curr_loc = path[-1]
        if board[curr_loc] == 2:
            return path, len(path)
        adj_list = find_adjacent(board, curr_loc, visited)
        visited.extend(adj_list)
        for adj in adj_list:
            new_path = list(path)
            new_path.append(adj)
            queue.append(new_path)
    return None


def DFS(board):
    start = np.where(board == 1)
    curr_loc = (start[0][0], start[1][0])

    visited = []
    stack = []
    stack.append([curr_loc])
    while stack:
        path = stack.pop(-1)
        curr_loc = path[-1]
        if board[curr_loc] == 2:
            return path, len(path)
        adj_list = find_adjacent(board, curr_loc, visited)
        visited.extend(adj_list)
        for adj in adj_list:
            new_path = list(path)
            new_path.append(adj)
            stack.append(new_path)
    return None


def A_star(board):
    start = np.where(board == 1)
    start_node = (start[0][0], start[1][0])
    heuristics = get_heuristics(board)

    came_from = {}
    open_list = [start_node]

    g_score = {}
    g_score[start_node] = 0
    f_score = {}
    f_score[start_node] = heuristics[start_node]

    f_score = np.zeros(board.shape, dtype=int) + 1

    while open_list:
        curr_node = min(open_list, key=lambda x: g_score[x[0], x[1]] + f_score[x[0], x[1]])
        if board[curr_node] == 2:
            return reconstruct_path(curr_node, came_from), len(reconstruct_path(curr_node, came_from))

        open_list.remove(curr_node)
        adj_list = find_adjacent(board, curr_node)
        for adj_node in adj_list:
            temp_g_score = g_score[curr_node] + 1
            if temp_g_score < g_score.get(adj_node, math.inf):
                came_from[adj_node] = curr_node
                g_score[adj_node] = temp_g_score
                f_score[adj_node] = g_score[adj_node] + heuristics[adj_node]
                if adj_node not in open_list:
                    open_list.append(adj_node)
    return None


def reconstruct_path(curr_node, came_from):
    total_path = [curr_node]
    while curr_node in came_from.keys():
        curr_node = came_from[curr_node]
        total_path.insert(0, curr_node)
    return total_path


def get_heuristics(board):
    end = np.where(board == 2)
    end = (end[0][0], end[1][0])
    heuristics_board = np.zeros(board.shape, dtype=int)

    for i, row in enumerate(board):
        for j, col in enumerate(row):
            heuristics_board[i, j] = abs(i - end[0]) + abs(j - end[1])
    return heuristics_board


obstacle_list = [(5, 6), (3, 4)]
start = (5, 23)
#start = (3,2)
end = (13, 2)
board = generate_board(obstacle_list, start, end)
print(BFS(board))
