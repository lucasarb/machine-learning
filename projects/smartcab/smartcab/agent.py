from __future__ import division
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import csv

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables 
        self.valid_directions = (None, 'forward', 'left', 'right')
        self.alpha = float(0.5) #learning rate
        self.gamma = float(0.5) #discount rate
        self.epsilon = float(0.1) #exploration rate
        self.q_table = dict() #table of Q-values
        self.hard_deadline = None
        self.successes = []
        self.time_taken = []
        self.optimal_route = []
        self.counter = Counter()
        self.n_trials = 0

        ##Populate q_table with 0
        light = ['red','green']
        oncoming = [None, 'forward', 'left', 'right']
        right = [None, 'forward', 'left', 'right']
        left = [None, 'forward', 'left', 'right']
        next_waypoint = ['forward', 'left', 'right']

        possible_actions = [None, 'forward', 'left', 'right']

        for li in light:
            for on in oncoming:
                for ri in right:
                    for le in left:
                        for ne in next_waypoint:
                            d_key = (li,on,ri,le,ne)
                            self.q_table[d_key] = dict.fromkeys(possible_actions,0)




    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        location = self.env.agent_states[self]['location']
        self.hard_deadline = self.env.get_deadline(self)
        self.optimal_route.append(abs(location[0] - destination[0]) +abs(location[1] - destination[1]))
        self.n_trials += 1


    def select_action(self):
        # Select the action according to the q values
        valid_directions = self.valid_directions
        best_2_action = sorted(self.q_table[self.state], key = self.q_table[self.state].get, reverse = True)[:2]
        epsilon = self.epsilon

        #Increase Exploration
        if random.random() <= epsilon:
            action = best_2_action[random.randint(0,1)]
        else:
            action = max(self.q_table[self.state], key = self.q_table[self.state].get)

        return action

    def update_q_value(self, reward, action):
        #Update the Q value after receiving the reward
        alpha = self.alpha
        gamma = self.gamma

        self.q_table[self.state][action] += alpha*(reward + gamma * max(self.q_table[self.state].values()) - self.q_table[self.state][action])


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        destination = self.env.agent_states[self]['destination']

        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'],inputs['right'],inputs['left'],self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.select_action()

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0:
            #count the penalties in each trial
            self.counter.update([self.n_trials])
            #global count of penalties
            self.counter.update(['Penalty'])
        else:
            self.counter.update(['Correct'])

        #update the location after the action
        location = self.env.agent_states[self]['location']

        #Update if it was a success and calculate the time it took to get to the destination
        if destination == location:
            self.successes.append(True)
            self.time_taken.append(self.hard_deadline - deadline)


        # TODO: Learn policy based on state, action, reward
        # Update q-table
        self.update_q_value(reward,action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    number_trials = 100
    optimal_search = [[0.5,0.8,0.1],[0.8,0.8,0.1],[0.3,0.8,0.1],[0.5,0.3,0.1],[0.5,0.5,0.05],[0.5,0.8,0.3],[0.5,0.8,0.05],[0.5,0.5,0.1],[0.8,0.5,0.05],[0.3,0.5,0.1]]
    #success_values = {}
    total_counter = {}



    for number in range(0,len(optimal_search)):

        print "Test number #{}: alpha, gamma, epsilon = {}".format(number,optimal_search[number])
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        # Now simulate it
        sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        #Change the parameters each iteration
        #a.alpha, a.gamma , a.epsilon = optimal_search[number][0],optimal_search[number][1],optimal_search[number][2]

        sim.run(n_trials=number_trials)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

        print "Successes: {}%".format(100*(len(a.successes)/number_trials))
        #success_values[number] = len(a.successes)/number_trials
        total_counter[number] = a.counter

    #Find the success rate in search for the optimal parameters
    #with open('data.csv','a') as file:
    #    w = csv.DictWriter(file, success_values.keys())
    #    w.writeheader()
    #    w.writerow(success_values)

    #Keep track of the policy penalties
    with open('data_penalties.csv', 'a') as file:
        w = csv.DictWriter(file, total_counter.keys())
        w.writeheader()
        w.writerow(total_counter)

    
if __name__ == '__main__':
    run()
