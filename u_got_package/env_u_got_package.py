import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from termcolor import colored, cprint

# Actions in the environment. Go up, down, left, right.
Action = namedtuple('Action', 'name index delta_y delta_x string_action')
# act = Action('act', 4, 0, 0)

# Station can be from: post_office, bank, council_office,gp, people.
# For each there is a reward when needed if you arrive to ta station for the first time and you have all perlimenery documents.
# There is another reward if you already recieved the service or you do not hold all perlimenrary documents.
# If the station is people the rewards are for not keeping social distancing of more than 1 cell (rewards are the same)
Station = namedtuple('Station', 'name, grid_number, reward_success, grid_position, grid_letter, grid_color')

# State are of form (position y, position x, documents you have) and more precisely:
# ('StateTuple', 'position_y, position_x, has_utility_bill, has_paid_customs, has_vacc_approval')
# position values are integer from 0-N size of environment and other values are 0/1 boolean values as a number)
StateTuple = namedtuple('StateTuple',
                        'position_y, position_x, has_utility_bill, has_paid_customs, has_vacc_approval, money')

up = Action('up', 0, -1, 0, 'UP')
down = Action('down', 1, +1, 0, 'DOWN')
left = Action('left', 2, 0, -1, 'LEFT')
right = Action('right', 3, 0, 1, 'RIGHT')


class U_GotPackage():
    """
        U Got Package Game
        Based on a famous Israeli game invented by Ephraim Kishon.
        See :https://boardgamegeek.com/boardgame/38411/package-has-arrived and about game inventor: https://en.wikipedia.org/wiki/Ephraim_Kishon

        Description:
        One day you receive a letter informing you that a package has arrived for you from your dear sister in Australia.
        In order to receive it you will have to come to the special post office in Dover port.
        When you arrive there you unfortunately realize that you are going to have to go though many buracracy stations in order to actualiy recieve your package.

        stations:
        Post office port
        Only if you arrive here with customs paid, utility bill you can get the package

        Bank branch :
        Here you can pay Customs payment
        you need to have utility bill before recieving the service.
        Sometimes when in a bad mood they can send you to bring also a vaccination approval


        Council office:
        Here you can get a new copy of the utility bill.
        Sometimes when in a bad mood they can send you to bring also a vaccination approval


        GP:
        Here you can receive a letter approving you have been vaccinated

        People:The
        You have to keep you social distancing from people and will have to keep at least 1 cell apart. otherwise get a bad reward.

        state:
        (position_y, position_x, has_utility_bill, has_paid_customs, has_vacc_approval)
        """

    def __init__(self, N=10, custom_pay=2):
        '''
        Constructor of u got package. Initzalization of attributes, Constant values, creation of game grid
        :param N: environemnt size. Will be N*N
        :param custom_pay: money amount needed to pay customs at the bank
        '''

        # env_dict=predefined_environments[env_type]
        # Initalization of constants
        self.N = N
        self.ACTIONS = [up, down, left, right]
        self.CUSTOM_PAY = custom_pay
        #Timesteps for a paycut(of 1) in the money from jobs
        self.TIMEUNITS_PAYCUT = 2 * N ** 2
        #Rewards are adjusted to each environemnt
        self.COUNCIL_OFFICE_REWARD = 3 * (N ** 2)
        self.BANK_REWARD = 6 * (N ** 2)
        self.GP_REWARD = (N ** 2)
        # self.TIMEUNITS_GAME_OVER= 20 * (self.N ** 2)
        #Fine for not keeping one cell apart form people
        self.PEOPLE_BUMP_REWARD = -1 * N // 2

        #Rewards for arriving to a station without correct documents or after recieving the service
        self.REWARD_TIME_WASTE = -N
        #Number of people in the environemnt
        self.NUM_PEOPLE_ALLOCATE = self.N // 4
        self.PROBABILITY_VACC_COUNCIL=0.7
        self.PROBABILITY_VACC_BANK=0.3
        self.TIME_LIMIT=20 * (self.N ** 2)

        #station values  -namedtuple('Station', 'name, grid_number, reward_success, grid_position, grid_letter, grid_color')
        self.stations_mapping = {'post office': Station('post office', 4, 0, (N - 1, N - 1), ' PO', 'magenta'),
                                 'bank': Station('bank', 3, self.BANK_REWARD, (N // 2, N // 2), ' BA', 'yellow'),
                                 'council office': Station('council office', 2, self.COUNCIL_OFFICE_REWARD, (0, 0),
                                                           'CO ', 'blue'),
                                 'gp': Station('gp', 5, self.GP_REWARD, (0, N - 1), ' GP', 'green'),
                                 'people': Station('people', 6, -1 * N, None, ' PP', 'red'),
                                 'job1': Station('job1', 7, 3, (N - 1, 0), 'JB1', 'cyan'),
                                 'job2': Station('job2', 8, 2, (N // 2, 0), 'JB2', 'cyan')}

        self.post_office = self.stations_mapping['post office']
        self.gp = self.stations_mapping['gp']
        self.bank = self.stations_mapping['bank']
        self.people = self.stations_mapping['people']
        self.council_office = self.stations_mapping['council office']
        self.job1 = self.stations_mapping['job1']
        self.job2 = self.stations_mapping['job2']

        self.num_actions = len(self.ACTIONS)
        self.MAX_SUM_PAY = self.job1.reward_success + self.job2.reward_success
        self.num_states = (self.N ** 2) * (2 ** 3) * (self.MAX_SUM_PAY + 1)  # grid size (N*N) positionsX 8 options for documents in a state, each 0/1* money value

        self.station_positions = []
        for station in self.stations_mapping.keys():
            if station != 'people':
                self.station_positions.append(self.stations_mapping[station].grid_position)

        #mapping between state index(int) and tuple representing the state namedtuple('StateTuple','position_y, position_x, has_utility_bill, has_paid_customs, has_vacc_approval, money')
        self.state_index2tuple, self.state_tuple2index = self.create_state_dictionaries()

        self.grid = self.create_env_grid()

        self.reset()


    def create_env_grid(self):
        '''
        Creating the grid world. allocation of stations on the grid
        :return: grid
        '''

        #numpy based grid
        self.grid = np.zeros((self.N, self.N))

        self.grid[self.post_office.grid_position] = self.post_office.grid_number
        self.grid[self.gp.grid_position] = self.gp.grid_number
        self.grid[self.council_office.grid_position] = self.council_office.grid_number
        self.grid[self.bank.grid_position] = self.bank.grid_number
        self.grid[self.job1.grid_position] = self.job1.grid_number
        self.grid[self.job2.grid_position] = self.job2.grid_number

        return self.grid

    def allocate_people(self):
        '''
        Allocate people on the grid. Look for empty location and not too close to station or other people for not blocking
        :return:
        '''

        # zero out the current people positions
        self.grid = np.where(self.grid != self.people.grid_number, self.grid, 0)

        people_positions = []

        # Allocate people position- searching for candidates- adding if not near station or another person
        for person_num in range(self.NUM_PEOPLE_ALLOCATE):
            found = False
            while not found:
                position_candidate = self.find_empty_position()
                good_candidate = True
                for station in self.station_positions:
                    if ((abs(position_candidate[0] - station[0]) <= 2) and (
                            abs(position_candidate[1] - station[1]) <= 2)):
                        good_candidate = False

                for people_p in people_positions:
                    if ((abs(position_candidate[0] - people_p[0]) <= 1) and (
                            abs(position_candidate[1] - people_p[1]) <= 1)):
                        good_candidate = False
                if good_candidate:
                    people_positions.append(position_candidate)
                    self.grid[position_candidate[0], position_candidate[1]] = self.people.grid_number
                    found = True


    def from_state_tuple_to_state(self, state_tuple: StateTuple):
        '''
        translate state tuple to a state index(int)
        :param state_tuple:
        :return: state_index (int)
        '''

        state_index = 0
        position_number = state_tuple.position_y * self.N + state_tuple.position_x

        num_states_max_pay = 1 + self.MAX_SUM_PAY
        # for each position there are 2**8 possibilities
        state_index += position_number * (8 * (num_states_max_pay)) #Blocks of 8*money. Number of blocks is N*N

        docs_index = state_tuple.has_utility_bill * (4 * num_states_max_pay) + state_tuple.has_paid_customs * (
                    2 * num_states_max_pay) + state_tuple.has_vacc_approval * num_states_max_pay  #In each block of grid, blocks document *max money

        money_index = state_tuple.money

        state_index += docs_index + money_index

        return state_index

    def create_state_dictionaries(self):
        '''
        Creating of dictionaries to translation of state_index to tuple state and vise versa
        :return: state_index2tuple, state_tuple2index
        '''

        state_index2tuple = {}
        state_tuple2index = {}

        for position_y in range(self.N):
            for position_x in range(self.N):
                for has_utility_bill in range(2):
                    for has_paid_customs in range(2):
                        for has_vac_approval in range(2):
                            for money in range(1 + self.MAX_SUM_PAY):
                                state_tuple = StateTuple(position_y, position_x, has_utility_bill, has_paid_customs,
                                                         has_vac_approval, money)
                                state_index = self.from_state_tuple_to_state(state_tuple)
                                state_index2tuple[state_index] = state_tuple
                                state_tuple2index[state_tuple] = state_index

        return state_index2tuple, state_tuple2index

    def step(self, action_index, render=False):
        '''
        Play one step in the game according to the action parameter
        :param action_index: action index (0-3) according to [up, down, left, right]
        :return: state_index- The new state index
                  reward
                  done- True/False if the game finished (finishes either got the package or run out ot timelimit)
                  got_package- True/False if achieved getting the package (used mainly for statistics)
        '''

        action = self.ACTIONS[action_index]
        #print(action)
        done = False
        reward = -1 #fine for every timestep passed
        stay = 0
        self.time += 1
        got_package = False

        #Current state values
        has_utility_bill = self.agent_state.has_utility_bill
        has_paid_customs = self.agent_state.has_paid_customs
        has_vacc_approval = self.agent_state.has_vacc_approval
        earned_until_now = self.agent_state.money

        #print("stepy:", self.agent_state.position_y)
        #print("stepx:", self.agent_state.position_x)

        #New positions
        new_position_y = self.agent_state.position_y + action.delta_y
        new_position_x = self.agent_state.position_x + action.delta_x

        #print("step y:", new_position_y)
        #print("step x:", new_position_x)
        #print(action.string_action)

        # If off the grid, stay in position and recieve penalty
        if ((new_position_y in [-1, self.N]) or (new_position_x in [-1, self.N])):
            reward += -1
            new_position_x = self.agent_state.position_x
            new_position_y = self.agent_state.position_y
            stay = 1

        if not stay:
            #Handle proximity to people- fine if agent it too near to pther people
            proximity = self.proximity_people(new_position_y, new_position_x)
            if proximity <= 1:
                reward += self.PEOPLE_BUMP_REWARD
            # Handle post office
            elif self.grid[new_position_y, new_position_x] == self.post_office.grid_number:
                if ((has_paid_customs) and (has_utility_bill)):
                    reward += self.post_office.reward_success
                    done = True
                    self.game_stats['done'] = self.time
                    self.game_stats['got package'] = self.time
                    got_package = True
                    if render:
                        print("Well done! Package is yours")
                else:
                    reward += self.REWARD_TIME_WASTE #If arrived without all documents recieve a penalty
            # Handle council office- Here we recieve utility bill
            elif self.grid[new_position_y, new_position_x] == self.council_office.grid_number:
                if has_utility_bill:
                    reward += self.REWARD_TIME_WASTE  #If already recieved the service get a fine
                elif ((not has_utility_bill) and (not has_vacc_approval)):
                    if np.random.rand(1)[0] < self.PROBABILITY_VACC_COUNCIL: #With probability are sent to get a vaccination certificate
                        reward += self.REWARD_TIME_WASTE
                    else:
                        reward += self.council_office.reward_success #Get utility bill and recieve a reward
                        has_utility_bill = 1
                        self.game_stats['has utility bill'] = self.time
                elif ((not has_utility_bill) and (has_vacc_approval)):
                    reward += self.council_office.reward_success
                    has_utility_bill = 1
                    self.game_stats['has utility bill'] = self.time
            # Handle bank- Here we can pay for customs
            elif self.grid[new_position_y, new_position_x] == self.bank.grid_number:
                if (((has_paid_customs) or (not has_utility_bill)) or (earned_until_now < self.CUSTOM_PAY)):
                    reward += self.REWARD_TIME_WASTE  #Recieve fine we have alredy paid customs or arrived without utility bill of not enough money
                elif (((not has_paid_customs) and (has_utility_bill)) and (earned_until_now >= self.CUSTOM_PAY)):
                    if not has_vacc_approval:
                        if np.random.rand(1)[0] < self.PROBABILITY_VACC_BANK: #With probability are sent to get a vaccination certificate
                            reward += self.REWARD_TIME_WASTE
                        else:
                            reward += self.bank.reward_success
                            has_paid_customs = 1  #Pay customs and recieve a reward
                            self.game_stats['paid customs'] = self.time
                    elif has_vacc_approval:
                        reward += self.bank.reward_success
                        has_paid_customs = 1
                        self.game_stats['paid customs'] = self.time
            #Handle gp- here the agent can get a vaccination certificate
            elif self.grid[new_position_y, new_position_x] == self.gp.grid_number:
                if has_vacc_approval:
                    reward += self.REWARD_TIME_WASTE
                elif not has_vacc_approval:
                    reward += self.gp.reward_success
                    self.game_stats['has vac approval'] = self.time
                    has_vacc_approval = 1
            #Handle jobs
            elif self.grid[new_position_y, new_position_x] == self.job1.grid_number:
                pay = self.jobs_pay['job1']
                self.jobs_pay['job1'] = 0 #After the job is done-zero out payment
                # reward += pay * (self.N )
                earned_until_now = earned_until_now + pay
                self.game_stats['money earned'] = earned_until_now
            elif self.grid[new_position_y, new_position_x] == self.job2.grid_number:
                pay = self.jobs_pay['job2']
                self.jobs_pay['job2'] = 0  #After the job is done-zero out payment
                # reward+=pay*(self.N)
                earned_until_now = earned_until_now + pay
                self.game_stats['money earned'] = earned_until_now

            if not self.time % self.TIMEUNITS_PAYCUT:  #Perform a timely payecut of -1
                self.jobs_pay['job1'] += (0 if self.jobs_pay['job1'] == 0 else -1)
                self.jobs_pay['job2'] += (0 if self.jobs_pay['job2'] == 0 else -1)
                if (earned_until_now + self.jobs_pay['job1'] + self.jobs_pay['job2']) < self.MAX_SUM_PAY:
                    done = True
                    if render:
                        print("Sorry! Agent failed. No potential of earning money to pay customs")

        self.agent_state = StateTuple(new_position_y, new_position_x, has_utility_bill, has_paid_customs,
                                      has_vacc_approval, earned_until_now)
        state_index = self.state_tuple2index[self.agent_state]

        self.visited_positions.append((self.agent_state.position_y, self.agent_state.position_x))

        if self.time >= self.TIME_LIMIT:
            done = True
            if render:
                print("Sorry! Passed time limit")

        return state_index, reward, done, got_package

    def reset(self):
        '''
        Reset the environemnt. zero time and money/ find random place to agent
        :return:
        '''
        self.time = 0
        self.game_stats = {'has utility bill': 0, 'paid customs': 0, 'has vac approval': 0, 'got package': 0,
                           'close proximity': 0, 'done': 0, 'money earned': 0}
        self.visited_positions = []
        # self.accident=self.change_accident_location()

        self.jobs_pay = {'job1': self.job1.reward_success, 'job2': self.job2.reward_success}
        self.allocate_people()

        agent_position = self.find_empty_position()

        self.agent_state = StateTuple(agent_position[0], agent_position[1], 0, 0, 0,
                                      0)  # StateTuple(self.N-1, 0, 0, 0,0)

        # observation = Observation(self.agent_state.position_y, self.agent_state.position_x, 0)
        # observation_index = self.from_observation_tuple_to_index(observation)
        state_index = self.from_state_tuple_to_state(self.agent_state)

        return state_index

    def find_empty_position(self):
        '''
        Find en ampty position in the grid
        :return: position as a list of two value for [y, x]
        '''

        position = np.random.randint(self.N, size=2)

        while self.grid[position[0], position[1]] > 0:
            #If it is a number above 0 it means it is Occupied.
            position = np.random.randint(self.N, size=2)

        return position

    def proximity_people(self, position_y, position_x):
        '''
        Return the min proximity to people from the input position
        :param position_y:
        :param position_x:
        :return:
        '''
        proximity = 3 #We want to minimise this . It will only matter if it is under 2 so 3 is fine
        people_positions = np.argwhere(self.grid == self.people.grid_number)
        all_proximity = []

        # To be close it mean  that  in one column or row the same as a person and in the other position less or equal to 1.
        for position in people_positions:
            if position_x == position[1]:
                prox_y = abs(position_y - position[0])
                if prox_y <= 1:
                    #It has to be at least in on column or row
                    all_proximity.append(prox_y)
            if position_y == position[0]:
                prox_x = abs(position_x - position[1])
                if prox_x <= 1:
                    all_proximity.append(prox_x)

        if len(all_proximity):
            proximity = min(all_proximity) #min proximity from the list

        return proximity

    def display2(self):
        '''
        display grid as a matplotlib
        :return:
        '''

        color_GP = ("r" if self.agent_state.has_vacc_approval else "b")
        color_BA = ("r" if self.agent_state.has_paid_customs else "b")
        color_CO = ("r" if self.agent_state.has_utility_bill else "b")

        image_grid = np.zeros((self.N, self.N))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        # ax = plt.gca()
        ax.set_xticks(np.arange(0.5, self.N, 1))
        ax.set_yticks(np.arange(0.5, self.N, 1))
        ax.set_xticklabels(np.arange(0, self.N))
        ax.set_yticklabels(np.arange(0, self.N))
        for position_y in range(self.grid.shape[0]):
            for position_x in range(self.grid.shape[1]):
                if self.grid[position_y, position_x] == self.bank.grid_number:
                    ax.text(position_x, position_y, self.bank.grid_letter, ha="center", va="center", color=color_BA)
                elif self.grid[position_y, position_x] == self.gp.grid_number:
                    ax.text(position_x, position_y, self.gp.grid_letter, ha="center", va="center", color=color_GP)
                elif self.grid[position_y, position_x] == self.council_office.grid_number:
                    ax.text(position_x, position_y, self.council_office.grid_letter, ha="center", va="center",
                            color=color_CO)
                elif self.grid[position_y, position_x] == self.post_office.grid_number:
                    ax.text(position_x, position_y, self.post_office.grid_letter, ha="center", va="center", color='b')
                elif self.grid[position_y, position_x] == self.people.grid_number:
                    ax.text(position_x, position_y, self.people.grid_letter, ha="center", va="center", color='w')
                    image_grid[position_y, position_x] = 1
                elif self.grid[position_y, position_x] == self.job1.grid_number:
                    ax.text(position_x, position_y, self.job1.grid_letter, ha="center", va="center", color='b')
                elif self.grid[position_y, position_x] == self.job2.grid_number:
                    ax.text(position_x, position_y, self.job2.grid_letter, ha="center", va="center", color='b')
                #    ax.text(position_x, position_y, self.lottery.grid_letter, ha="center", va="center", color='b')
                # elif self.grid[position_y, position_x]==self.wall.grid_number:

        # ax.grid(color='b', linewidth=2, axis='both')
        # ax.grid()
        ax.imshow(image_grid, cmap='BuGn')  # , cmap='BuGn'
        ax.grid()
        ax.set_frame_on(False)
        plt.show()
        print('Agent state:')
        print(f'Position X: {self.agent_state.position_x} Position Y: {self.agent_state.position_y}')
        print(
            f'Has utility bill: {bool(self.agent_state.has_utility_bill)} Has paid customs: {bool(self.agent_state.has_paid_customs)} ')
        print(f'Has vacc approval: {bool(self.agent_state.has_vacc_approval)}')

    def map_display(self, position_y, position_x):
        #'''
        #This funtion serves the function display (the next function). maps a position to a color and letter
        #:param position_y:
        #:param position_x:
        #:return: (letter, color) mapping of the input position
        #'''

        #maps a position to a color and letter
        if self.grid[position_y, position_x] == self.bank.grid_number:
            return self.bank.grid_letter, self.bank.grid_color
        elif self.grid[position_y, position_x] == self.gp.grid_number:
            return self.gp.grid_letter, self.gp.grid_color
        elif self.grid[position_y, position_x] == self.post_office.grid_number:
            return self.post_office.grid_letter, self.post_office.grid_color
        elif self.grid[position_y, position_x] == self.council_office.grid_number:
            return self.council_office.grid_letter, self.council_office.grid_color
        elif self.grid[position_y, position_x] == self.job1.grid_number:
            return self.job1.grid_letter, self.job1.grid_color
        elif self.grid[position_y, position_x] == self.job2.grid_number:
            return self.job2.grid_letter, self.job2.grid_color
        elif self.grid[position_y, position_x] == self.people.grid_number:
            return self.people.grid_letter, self.people.grid_color
        else:
            return ' | ', 'grey'

    def display(self, action_index=None):
        '''
        Main display function. display the grid as a printing with color for stations and a marker for current poistion of the agent
        :param action_index: Showing a prinign of this input action. Can be used to display the next action in order to understand the running of the game
        :return:
        '''



        envir_display = self.grid.copy()
        envir_display[self.agent_state.position_y, self.agent_state.position_x] = 1
        line = "=  "
        for start in range(self.N):
            line += ' = '
        line += " =\n"

        for row in range(self.N):
            line += "=  "
            for col in range(self.N):
                attr = []
                if (self.agent_state.position_y == row and self.agent_state.position_x == col):
                    attr = ["reverse", "blink"]

                letter, color = self.map_display(row, col)
                # print(letter, color)
                line += colored(letter, color, attrs=attr)
            line += " =\n"

        line += "=  "
        for start in range(self.N):
            line += ' = '
        line += " =\n"

        print(line)

        if action_index is not None:
            action = self.ACTIONS[action_index]
            print("Action:" + action.string_action)

        print(
            f'UTILITY: {bool(self.agent_state.has_utility_bill)} PAID CUSTOMS: {bool(self.agent_state.has_paid_customs)} ')
        print(f'VACC APPROVAL: {bool(self.agent_state.has_vacc_approval)} MONEY_EARNED: {self.agent_state.money}')


    def translate_q_values_to_position_grid(self, q_values_matrix):
        '''

        :param q_values_matrix:
        :return:
        '''

        q_values_grid = np.zeros((self.N, self.N))

        for state_index in range(q_values_matrix.shape[0]):
            state = self.state_index2tuple[state_index]
            q_values_grid[state.position_y, state.position_x] = max(max(q_values_matrix[state_index, :]),
                                                                    q_values_grid[state.position_y, state.position_x])

        # with np.printoptions(precision=1):
        #    print(f"---------------------------------------")
        #    print(q_values_grid)

        return q_values_grid
