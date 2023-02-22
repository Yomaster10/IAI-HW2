import time

import Gobblet_Gobblers_Env as gge
import submission

# TODO - elaborate
time_limit = 1
steps_limit = 100

agents = {
    "human": submission.human_agent,
    "random": submission.random_agent,
    "greedy": submission.greedy,
    "greedy_improved": submission.greedy_improved,
    "minimax": submission.rb_heuristic_min_max,
    "alpha_beta": submission.alpha_beta,
    "expectimax": submission.expectimax
}


# gets two functions of the agents and plays according to their selection
# plays the game and returns 1 if agent1 won and -1 if agent2 won and 0 if there is a tie
def play_game(agent_1_str, agent_2_str):
    agent_1 = agents[agent_1_str]
    agent_2 = agents[agent_2_str]
    s = gge.State()
    env = gge.GridWorldEnv('human')
    winner = None
    env.reset()
    env.render()
    start_time = 0
    end_time = 0
    steps_per_game = 0
    while winner is None:
        if env.s.turn == 0:
            print("player 0")
            start_time = time.time()
            chosen_step = agent_1(env.get_state(), 0, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit and (agent_1_str in ["minimax", "alpha_beta", "expectimax"]):
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)
        else:
            print("player 1")
            start_time = time.time()
            chosen_step = agent_2(env.get_state(), 1, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit and (agent_2_str in ["minimax", "alpha_beta", "expectimax"]):
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)

        s = env.get_state()
        winner = gge.is_final_state(env.s)
        if steps_per_game >= steps_limit:
            winner = 0
    if winner == 0:
        print("tie")
    else:
        print("winner is:", winner)
    return winner


# plays many games between two agents and returns in percentage win per player and ties
def play_tournament(agent_1_str, agent_2_str, num_of_games):
    # agent_1 = agents[agent_1_str]
    # agent_2 = agents[agent_2_str]
    # score is [ties, wins for agent1, wins for agent2]
    score = [0, 0, 0]

    for i in range(num_of_games):
        tmp_score = int(play_game(agent_1_str, agent_2_str))
        score[int(tmp_score)] = score[int(tmp_score)] + 1

    for j in range(num_of_games):
        tmp_score = int(play_game(agent_2_str, agent_1_str))
        real_tmp_score = 0
        if tmp_score != 0:
            if tmp_score == 1:
                real_tmp_score = 2
            else:
                real_tmp_score = 1
        score[int(real_tmp_score)] = score[int(real_tmp_score)] + 1


    print("ties: ", (score[0] / (num_of_games*2)) * 100, "% ", agent_1_str, "player1 wins: ", (score[1] / (num_of_games*2)) * 100,
          "% ", agent_2_str, "player2 wins: ", (score[2] / (num_of_games*2)) * 100)

    print("")
