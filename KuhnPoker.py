import numpy as np
import time as t
import sys as s
from random import shuffle

class Node:
    """
    Initializes a Node with a key, a set of actions, and the number of actions
    """
    def __init__(self, key, actions, numActions=2):
        self.key = key
        self.actions = actions
        self.numActions = numActions
        self.sumReach = 0
        self.reach = 0
        self.sumRegret = np.zeros(self.numActions)
        self.sumPolicy = np.zeros(self.numActions)
        self.policy = np.repeat(1/self.numActions, self.numActions)
    
    """
    String representation of a node
    """
    def __str__(self):
        avgStrategies = [f"{x:03.2f}" for x in self.get_avg_policy()]
        formattedKey = self.key.ljust(6)

        return f"{formattedKey} {avgStrategies}"
    
    """
    Update the policy based on the current reach and policy
    """
    def update_policy(self):
        self.sumPolicy += self.reach * self.policy
        self.sumReach += self.reach
        self.policy = self.get_policy()
        self.reach = 0
    
    """
    Get new policy based on the sum of regrets
    """
    def get_policy(self):
        regrets = self.sumRegret.copy()
        regrets[regrets < 0] = 0
        standardizingSum = regrets.sum()
        if standardizingSum > 0:
            return regrets/standardizingSum
        else:
            return np.repeat(1/self.numActions, self.numActions)
    
    """
    Get average policy over reach
    """
    def get_avg_policy(self):
        policy = self.sumPolicy/self.sumReach
        sumPolicy = sum(policy)
        policy /= sumPolicy

        return policy

"""
A poker agent that implements the Counterfactual Regret Minimization (CFR)
algorithm to find the Nash Equilibrium strategy in Kuhn Poker.
"""
class KuhnPokerAgent:
    def __init__(self):
        self.nodes = {}
        self.expectedValue = 0
        self.nashEquilibrium = dict()
        self.currPlayer = 0
        self.numCards = 3
        # 0 - Q, 1 - K, 2 - A
        self.cardDeck = np.array([0, 1, 2])
        self.numActions = 2
    
    """
    Trains the agent using the CFR algorithm over a given number of iterations
    """
    def train(self, numIterations=50000):
        expectedValue = 0
        for _ in range(numIterations):
            shuffle(self.cardDeck)
            expectedValue += self.cfr("", 1, 1)
            for _, value in self.nodes.items():
                value.update_policy()
        
        expectedValue /= numIterations
        to_string(expectedValue, self.nodes)
    
    """
    Implementation of Counterfactual Regret Minimization (CFR)
    """
    def cfr(self, movesSoFar, heroRegret, villanRegret):
        isHero = len(movesSoFar) % 2 == 0
        if isHero:
            card = self.cardDeck[0]
        else:
            card = self.cardDeck[1]

        if self.is_terminal_state(movesSoFar):
            if isHero:
                heroCard = self.cardDeck[0]
                villanCard = self.cardDeck[1]
            else:
                heroCard = self.cardDeck[1]
                villanCard = self.cardDeck[0]
            utility = self.get_utility(movesSoFar, heroCard, villanCard)
            return utility
        
        value = self.get_node(card, movesSoFar)
        policy = value.policy
        actionUtility = np.zeros(self.numActions)

        for action in range(self.numActions):
            newMovesSoFar = movesSoFar + value.actions[action]
            if isHero:
                actionUtility[action] = -1 * self.cfr(newMovesSoFar, heroRegret * policy[action], villanRegret)
            else:
                actionUtility[action] = -1 * self.cfr(newMovesSoFar, heroRegret, villanRegret * policy[action])
        
        utility = sum(actionUtility * policy)
        regret = actionUtility - utility
        if isHero:
            value.reach += heroRegret
            value.sumRegret += villanRegret * regret
        else:
            value.reach += villanRegret
            value.sumRegret += heroRegret * regret
        
        return utility

    """
    Retrieves or creates a node based on the current game state
    """
    def get_node(self, card, movesSoFar):
        key = str(card) + " " + movesSoFar
        if key not in self.nodes:
            moves = {0: "p", 1: "b"}
            value = Node(key, moves)
            self.nodes[key] = value
            return value
        return self.nodes[key]

    """
    Checks if the current state is a terminal state
    """
    @staticmethod
    def is_terminal_state(movesSoFar):
        if movesSoFar[-2:] == "bp" or movesSoFar[-2:] == "pp" or movesSoFar[-2:] == "bb":
            return True
    
    """
    Returns the utility of the terminal state for the hero player
    """
    @staticmethod
    def get_utility(movesSoFar, heroCard, villanCard):
        if movesSoFar[-1] == "p":
            if movesSoFar[-2:] == "pp":
                return 1 if heroCard > villanCard else -1
            else:
                return 1
        elif movesSoFar[-2:] == "bb":
            return 2 if heroCard > villanCard else -2

"""
Prints the expected values for each player and the optimal policy
"""
def to_string(expectedValue, inputMap):
    # Sort the items of the input map (i_map) by their keys
    sortedList = sorted(inputMap.items(), key=lambda x: x[0])
    # Print the expected value for the players
    print(f"Expected value for Hero: {expectedValue}")

    print("\Optimal policy for Hero\n")
    # Iterate over items where the key length is even
    for _, value in filter(lambda x: len(x[0]) % 2 == 0, sortedList):
        print(value)

    print(f"\n\nExpected value for Villan: {-1*expectedValue}")

    print("\Optimal policy for Villan\n")
    # Iterate over items where the key length is odd
    for _, value in filter(lambda x: len(x[0]) % 2 == 1, sortedList):
        print(value)

if __name__ == "__main__":
    time = t.time()
    agent = KuhnPokerAgent()
    agent.train(100000)