import random

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def smart_heuristic(state, agent_id):
    rows_count1 = np.array([0, 0, 0])
    columns_count1 = np.array([0, 0, 0])
    diagonals_count1 = np.array([0, 0, 0])
    sum_two_together_1 = 0
    rows_count2 = np.array([0, 0, 0])
    columns_count2 = np.array([0, 0, 0])
    diagonals_count2 = np.array([0, 0, 0])
    sum_two_together_2 = 0
    for key, value in state.player1_pawns.items():
        if not np.array_equal(value[0], not_on_board) and not is_hidden(state, 0, key):
            row = value[0][0]
            column = value[0][1]
            rows_count1[row] += 1
            columns_count1[column] += 1
            if row == column:
                diagonals_count1[0] += 1
            if row + column == 2:
                diagonals_count1[1] += 1
    for key, value in state.player2_pawns.items():
        if not np.array_equal(value[0], not_on_board) and not is_hidden(state, 1, key):
            row = value[0][0]
            column = value[0][1]
            rows_count2[row] += 1
            columns_count2[column] += 1
            if row == column:
                diagonals_count2[0] += 1
            if row + column == 2:
                diagonals_count2[1] += 1
    
    is_final = gge.is_final_state(state)
    if is_final is not None:
        if is_final == 0:
            return 0
        winner = int(is_final) - 1
        if winner == agent_id:
            return 10
        else:
            return -10
                
    for row in rows_count1:
        if row == 2:
            sum_two_together_1 += 1
    for column in columns_count1:
        if column == 2:
            sum_two_together_1 += 1
    for diagonal in diagonals_count1:
        if diagonal == 2:
            sum_two_together_1 += 1
            
    for row in rows_count2:
        if row == 2:
            sum_two_together_2 += 1
    for column in columns_count2:
        if column == 2:
            sum_two_together_2 += 1
    for diagonal in diagonals_count2:
        if diagonal == 2:
            sum_two_together_2 += 1

    if agent_id == 0:
        return sum_two_together_1 - sum_two_together_2
    return sum_two_together_2 - sum_two_together_1


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -10
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def alpha_beta(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    raise NotImplementedError()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
