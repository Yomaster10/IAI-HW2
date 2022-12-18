import gym
import numpy as np
import pygame
from colorama import Fore
from gym import spaces
from gym.envs.registration import register

not_on_board = np.array([-1, -1])

# for internal use - converts the integer to the coordinates of the location
action_to_direction = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([0, 2]),
    3: np.array([1, 0]),
    4: np.array([1, 1]),
    5: np.array([1, 2]),
    6: np.array([2, 0]),
    7: np.array([2, 1]),
    8: np.array([2, 2]),
}

'''
size_cmp gets two possible sizes of pawns meaning B or M or S
and returns
 1 if size1 > size2
-1 if size2 > size1
0 if size1 == size2 
'''


def size_cmp(size1, size2):
    if size1 == size2:
        return 0
    if size1 == "B":
        return 1
    if size1 == "S":
        return -1
    if size2 == "S":
        return 1
    else:
        return -1


# for very internal use
def pawn_list_to_marks_array(curr_state):
    b_a = np.full((3, 3), " ")
    # all pawns of player 1

    # adding small pawns
    for key, pawn in curr_state.player1_pawns.items():
        if pawn[1] == "S":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "1"

    for key, pawn in curr_state.player2_pawns.items():
        if pawn[1] == "S":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "2"

    # adding medium pawns
    for key, pawn in curr_state.player1_pawns.items():
        if pawn[1] == "M":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "1"

    for key, pawn in curr_state.player2_pawns.items():
        if pawn[1] == "M":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "2"

    # adding big pawns
    for key, pawn in curr_state.player1_pawns.items():
        if pawn[1] == "B":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "1"

    for key, pawn in curr_state.player2_pawns.items():
        if pawn[1] == "B":
            if not np.array_equal(pawn[0], not_on_board):
                b_a[pawn[0][0], pawn[0][1]] = "2"

    return b_a


'''
find_curr_location is a function that gets the state,  a pawn, and the player of whom the pawn belongs to 
and returns the location on the board (as a tuple of coordinates)
'''


def find_curr_location(curr_state, pawn, player):
    if player is 0:
        for pawn_key, pawn_value in curr_state.player1_pawns.items():
            if pawn_key == pawn:
                return pawn_value[0]
    else:
        for pawn_key, pawn_value in curr_state.player2_pawns.items():
            if pawn_key == pawn:
                return pawn_value[0]


def is_legal_step(action, curr_state):
    pawn_list = {
        "agent1_big1": [curr_state.player1_pawns["B1"][0], "B"],
        "agent1_big2": [curr_state.player1_pawns["B2"][0], "B"],
        "agent1_medium1": [curr_state.player1_pawns["M1"][0], "M"],
        "agent1_medium2": [curr_state.player1_pawns["M2"][0], "M"],
        "agent1_small1": [curr_state.player1_pawns["S1"][0], "S"],
        "agent1_small2": [curr_state.player1_pawns["S2"][0], "S"],
        "agent2_big1": [curr_state.player2_pawns["B1"][0], "B"],
        "agent2_big2": [curr_state.player2_pawns["B2"][0], "B"],
        "agent2_medium1": [curr_state.player2_pawns["M1"][0], "M"],
        "agent2_medium2": [curr_state.player2_pawns["M2"][0], "M"],
        "agent2_small1": [curr_state.player2_pawns["S1"][0], "S"],
        "agent2_small2": [curr_state.player2_pawns["S2"][0], "S"]
    }

    if len(action[0]) != 2:
        print("ILLEGAL pawn type")
        return False
    if len(str(action[1])) != 1:
        print("ILLEGAL location type")
        return False
    if action[1] > 8 or action[1] < 0:
        print("ILLEGAL location type")
        return False

    if action[0][0] != "B" and action[0][0] != "M" and action[0][0] != "S":
        print("ILLEGAL pawn type")
        return False
    if action[0][1] != "1" and action[0][1] != "2":
        print("ILLEGAL pawn type")
        return False
    # checks if there is an attempt to place pawn on smaller pawn
    location = action_to_direction[action[1]]
    for key, value in pawn_list.items():
        if np.array_equal(value[0], location):
            if size_cmp(value[1], action[0][0]) >= 0:
                print("ILLEGAL placement of pawn")
                return False

    # finding current location
    curr_location = find_curr_location(curr_state, action[0], curr_state.turn)

    # check that the pawn is not under another pawn - relevant only to small and medium
    if action[0][0] != "B" and not np.array_equal(curr_location, not_on_board):
        for key, value in pawn_list.items():
            if np.array_equal(value[0], curr_location):
                if size_cmp(value[1], action[0][0]) > 0:
                    print("ILLEGAL pawn selection")
                    return False
    return True


'''
State is a class that represents the current state of the environment - more about that in the pdf
'''


class State:

    def __init__(self):
        self.turn = 0
        self.player1_pawns = {
            "B1": (not_on_board, "B"),  # change to name out - const np.array[-1,-1]
            "B2": (not_on_board, "B"),
            "M1": (not_on_board, "M"),
            "M2": (not_on_board, "M"),
            "S1": (not_on_board, "S"),
            "S2": (not_on_board, "S")
        }
        self.player2_pawns = {
            "B1": (not_on_board, "B"),
            "B2": (not_on_board, "B"),
            "M1": (not_on_board, "M"),
            "M2": (not_on_board, "M"),
            "S1": (not_on_board, "S"),
            "S2": (not_on_board, "S")
        }

    def insert_copy(self, new_state):
        self.turn = new_state.turn
        for key, value in new_state.player1_pawns.items():
            self.player1_pawns[key] = value
        for key, value in new_state.player2_pawns.items():
            self.player2_pawns[key] = value

    '''
    get_neighbors
    gets a parameter state of type State 
    returns a list of the neighbors each element is a tuple (action, State)
    where actions the action we did to reach to state from our state
    '''

    def get_neighbors(self):
        # TODO remember to check if dry_action returned None
        neighbor_list = []
        # all the pawns we can select
        pawns = ["B1", "B2", "M1", "M2", "S1", "S2"]
        # locations (it's just 0 to 8  so I will use a simple loop)
        for i in range(9):
            for pawn in pawns:
                next_state = State()
                next_state.insert_copy(self)
                # tmp_neighbor = self.dry_step((pawn, i), state)
                action = (pawn, i)
                if not is_legal_step(action, self):
                    continue
                if self.turn == 0:
                    next_state.player1_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])
                else:
                    next_state.player2_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])

                next_state.turn = (next_state.turn + 1) % 2
                neighbor_list.append((action, next_state))

        return neighbor_list


'''
is_final_state
gets a parameter curr_state of type State
and returns None if state isn't final
returns the winner: 1 if the first player won, 2 if the second player won, 0 if there is a tie 
'''


def is_final_state(curr_state):
    # use the array from render
    arr = pawn_list_to_marks_array(curr_state)
    win = None
    # check rows
    for i in range(3):
        if arr[i][0] == arr[i][1] and arr[i][2] == arr[i][1] and (not arr[i][0] == " ") and (
                not arr[i][1] == " ") and (not arr[i][2] == " "):
            if win is None:
                win = arr[i][0]
            else:
                if win != arr[i][0]:
                    return 0

    # check columns
    for j in range(3):
        if arr[0][j] == arr[1][j] and arr[2][j] == arr[1][j] and (not arr[0][j] == " ") and (
                not arr[1][j] == " ") and (not arr[2][j] == " "):
            if win is None:
                win = arr[0][j]
            else:
                if win != arr[0][j]:
                    return 0

    # check obliques
    if arr[0][0] == arr[1][1] and arr[2][2] == arr[1][1] and (not arr[0][0] == " ") and (not arr[1][1] == " ") and (
            not arr[2][2] == " "):
        if win is None:
            win = arr[0][0]
        else:
            if win != arr[0][0]:
                return 0

    if arr[0][2] == arr[1][1] and arr[2][0] == arr[1][1] and (not arr[0][2] == " ") and (not arr[1][1] == " ") and (
            not arr[0][2] == " "):
        if win is None:
            win = arr[1][1]
        else:
            if win != arr[1][1]:
                return 0

    if win is 1:
        return 1
    if win is 2:
        return 2
    return win


'''
render_console
gets a parameter curr_state of type State
prints the curr_state board to the console 
'''


def render_console(curr_state):
    # creates an array that specifies where the pawns on the board are
    # b_a is short for board array
    b_a = np.full((3, 3), "\033[0;36m B \033[1;37m")
    b_a.fill("  ")

    # creates description of the pawns that haven't been used (the ones on the side)
    # all pawns of player 1
    agent1_S1 = "  "
    if np.array_equal(curr_state.player1_pawns["S1"][0], not_on_board):
        agent1_S1 = Fore.CYAN + "S1" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["S1"][0][0], curr_state.player1_pawns["S1"][0][
            1]] = Fore.CYAN + "S1" + Fore.WHITE

    agent1_S2 = "  "
    if np.array_equal(curr_state.player1_pawns["S2"][0], not_on_board):
        agent1_S2 = Fore.CYAN + "S2" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["S2"][0][0], curr_state.player1_pawns["S2"][0][
            1]] = Fore.CYAN + "S2" + Fore.WHITE

    agent2_S1 = "  "
    if np.array_equal(curr_state.player2_pawns["S1"][0], not_on_board):
        agent2_S1 = Fore.LIGHTYELLOW_EX + "S1" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["S1"][0][0], curr_state.player2_pawns["S1"][0][
            1]] = Fore.LIGHTYELLOW_EX + "S1" + Fore.WHITE
    agent2_S2 = "  "
    if np.array_equal(curr_state.player2_pawns["S2"][0], not_on_board):
        agent2_S2 = Fore.LIGHTYELLOW_EX + "S2" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["S2"][0][0], curr_state.player2_pawns["S2"][0][
            1]] = Fore.LIGHTYELLOW_EX + "S2" + Fore.WHITE

    agent1_M1 = "  "
    if np.array_equal(curr_state.player1_pawns["M1"][0], not_on_board):
        agent1_M1 = Fore.CYAN + "M1" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["M1"][0][0], curr_state.player1_pawns["M1"][0][
            1]] = Fore.CYAN + "M1" + Fore.WHITE
    agent1_M2 = "  "
    if np.array_equal(curr_state.player1_pawns["M2"][0], not_on_board):
        agent1_M2 = Fore.CYAN + "M2" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["M2"][0][0], curr_state.player1_pawns["M2"][0][
            1]] = Fore.CYAN + "M2" + Fore.WHITE

    agent2_M1 = "  "
    if np.array_equal(curr_state.player2_pawns["M1"][0], not_on_board):
        agent2_M1 = Fore.LIGHTYELLOW_EX + "M1" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["M1"][0][0], curr_state.player2_pawns["M1"][0][
            1]] = Fore.LIGHTYELLOW_EX + "M1" + Fore.WHITE
    agent2_M2 = "  "
    if np.array_equal(curr_state.player2_pawns["M2"][0], not_on_board):
        agent2_M2 = Fore.LIGHTYELLOW_EX + "M2" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["M2"][0][0], curr_state.player2_pawns["M2"][0][
            1]] = Fore.LIGHTYELLOW_EX + "M2" + Fore.WHITE

    agent1_B1 = "  "
    if np.array_equal(curr_state.player1_pawns["B1"][0], not_on_board):
        agent1_B1 = Fore.CYAN + "B1" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["B1"][0][0], curr_state.player1_pawns["B1"][0][
            1]] = Fore.CYAN + "B1" + Fore.WHITE
    agent1_B2 = "  "
    if np.array_equal(curr_state.player1_pawns["B2"][0], not_on_board):
        agent1_B2 = Fore.CYAN + "B2" + Fore.WHITE
    else:
        b_a[curr_state.player1_pawns["B2"][0][0], curr_state.player1_pawns["B2"][0][1]] = Fore.CYAN + "B2" + Fore.WHITE

    # all pawns of player 2

    agent2_B1 = "  "
    if np.array_equal(curr_state.player2_pawns["B1"][0], not_on_board):
        agent2_B1 = Fore.LIGHTYELLOW_EX + "B1" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["B1"][0][0], curr_state.player2_pawns["B1"][0][
            1]] = Fore.LIGHTYELLOW_EX + "B1" + Fore.WHITE
    agent2_B2 = "  "
    if np.array_equal(curr_state.player2_pawns["B2"][0], not_on_board):
        agent2_B2 = Fore.LIGHTYELLOW_EX + "B2" + Fore.WHITE
    else:
        b_a[curr_state.player2_pawns["B2"][0][0], curr_state.player2_pawns["B2"][0][
            1]] = Fore.LIGHTYELLOW_EX + "B2" + Fore.WHITE

    print("           +------+------+------+      ")
    print(" ", agent1_B1, " ", agent1_B2, " | ", b_a[0, 0], " | ", b_a[0, 1], " | ", b_a[0, 2], " | ", agent2_B1,
          " ", agent2_B2, " ")
    print(" ", agent1_M1, " ", agent1_M2, " +------+------+------+ ", agent2_M1, " ", agent2_M2)
    print(" ", agent1_S1, " ", agent1_S2, " | ", b_a[1, 0], " | ", b_a[1, 1], " | ", b_a[1, 2], " | ", agent2_S1,
          " ", agent2_S2, " ")
    print("           +------+------+------+      ")
    print("           | ", b_a[2, 0], " | ", b_a[2, 1], " | ", b_a[2, 2], " | ")
    print("           +------+------+------+      ")


'''
cor_to_num
get coordination in the matrix and returns a number between 0 to 8 representing the location - like in the pdf
'''


def cor_to_num(cor):
    return 3 * cor[0] + cor[1]


'''
GridWorldEnv
the class that describes the environment and contains relevant functions 
'''


class GridWorldEnv(gym.Env):
    # internal definition - shouldn't concern you
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=3):
        self.size = size  # The size of the square grid
        self.window_size = 700  # The size of the PyGame window

        self.s = State()  # TODO - remove at the end maybe

        # we also need to chose the pawn but here we will regard only the position as if we have selected a pawn
        self.action_space = spaces.Discrete(9)

        # gets number and returns the coordination on the board

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # don't remove it's important
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.s = State()

    def _render_frame(self, curr_state):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Finally, add some gridlines
        # horizontal lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (200, 100 + (x + 1) * 100),
                (500, 100 + (x + 1) * 100),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (100 + (x + 1) * 100, 200),
                (100 + (x + 1) * 100, 500),
                width=3,
            )

            b_a = np.full((3, 3), "\033[0;36m B \033[1;37m")
            b_a.fill(" ")

            num_location_to_pixel = {
                0: (215, 210),
                1: (315, 210),
                2: (415, 210),
                3: (215, 310),
                4: (315, 310),
                5: (415, 310),
                6: (215, 410),
                7: (315, 410),
                8: (415, 410)
            }
            orange = pygame.image.load("images/orange.png").convert()
            blue = pygame.image.load("images/blue.png").convert()

            agent1_S1 = " "
            if np.array_equal(curr_state.player1_pawns["S1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (40, 56)), (23, 390))
            else:
                canvas.blit(pygame.transform.scale(blue, (40, 56)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["S1"][0])])
            agent1_S2 = " "
            if np.array_equal(curr_state.player1_pawns["S2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (40, 56)), (107, 390))
            else:
                canvas.blit(pygame.transform.scale(blue, (40, 56)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["S2"][0])])

            agent2_S1 = " "
            if np.array_equal(curr_state.player2_pawns["S1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (40, 56)), (547, 390))
            else:
                canvas.blit(pygame.transform.scale(orange, (40, 56)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["S1"][0])])
            agent2_S2 = " "
            if np.array_equal(curr_state.player2_pawns["S2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (40, 56)), (620, 390))
            else:
                canvas.blit(pygame.transform.scale(orange, (40, 56)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["S2"][0])])

            agent1_M1 = " "
            if np.array_equal(curr_state.player1_pawns["M1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (50, 70)), (20, 300))
            else:
                canvas.blit(pygame.transform.scale(blue, (50, 70)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["M1"][0])])
            agent1_M2 = " "
            if np.array_equal(curr_state.player1_pawns["M2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (50, 70)), (100, 300))
            else:
                canvas.blit(pygame.transform.scale(blue, (50, 70)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["M2"][0])])

            agent2_M1 = " "
            if np.array_equal(curr_state.player2_pawns["M1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (50, 70)), (543, 300))
            else:
                canvas.blit(pygame.transform.scale(orange, (50, 70)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["M1"][0])])
            agent2_M2 = " "
            if np.array_equal(curr_state.player2_pawns["M2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (50, 70)), (620, 300))
            else:
                canvas.blit(pygame.transform.scale(orange, (50, 70)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["M2"][0])])

            agent1_B1 = " "
            if np.array_equal(curr_state.player1_pawns["B1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (60, 84)), (20, 200))
            else:
                canvas.blit(pygame.transform.scale(blue, (60, 84)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["B1"][0])])
            agent1_B2 = " "
            if np.array_equal(curr_state.player1_pawns["B2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(blue, (60, 84)), (95, 200))
            else:
                canvas.blit(pygame.transform.scale(blue, (60, 84)),
                            num_location_to_pixel[cor_to_num(self.s.player1_pawns["B2"][0])])

            agent2_B1 = " "
            if np.array_equal(curr_state.player2_pawns["B1"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (65, 87)), (535, 200))
            else:
                canvas.blit(pygame.transform.scale(orange, (65, 87)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["B1"][0])])
            agent2_B2 = " "
            if np.array_equal(curr_state.player2_pawns["B2"][0], not_on_board):
                canvas.blit(pygame.transform.scale(orange, (65, 87)), (610, 200))
            else:
                canvas.blit(pygame.transform.scale(orange, (65, 87)),
                            num_location_to_pixel[cor_to_num(self.s.player2_pawns["B2"][0])])

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    '''
    get state
    returns the state of the environment in the struct of State  
    '''

    def get_state(self):
        return self.s

    '''
    step
    step gets a parameter action which is a tuple of (pawn, location)
    the function preforms the chosen action on the environment  
    '''

    def step(self, action):
        # checks that the step is legal
        if not is_legal_step(action, self.s):
            return

        if self.s.turn == 0:
            self.s.player1_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])
        else:
            self.s.player2_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])

        # advance the turn
        self.s.turn = (self.s.turn + 1) % 2
        return

    '''
    render
    prints the environment both to the console and both in a pygame window
    '''

    def render(self):
        render_console(self.s)
        return self._render_frame(self.s)

    # internal function
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    # these function registers the environment somewhere - internal
    register(
        id='gym_examples/GridWorld-v0',
        entry_point='gym_examples.envs:GridWorldEnv',
        max_episode_steps=300,
    )
