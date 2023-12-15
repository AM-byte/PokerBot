import random
import numpy as np
from numpy.random import choice

# A class that implements rock paper scissors 
class rock_paper_scissors():

    def __init__(self):
        # the number of actions as well as a list of possible actions
        # rock, paper , scissors
        self.actions = [0,1,2]
        self.num_actions = 3

        # ------------- needed only for the training method, everything else is passed as a var
        # Keeping track of the total regret of each decision
        # initially the regrets of rock, paper and scissors are all 0.
        # this is in the order of rock, paper, scissors
        self.regrets = [0,0,0]
        # The strategy holds the frequency the bot should play each action
        self.strategy = [0,0,0]

        # we need to store the opponents strategy and regrets as well to combat against their strategy
        self.oppregrets = [0,0,0]
        self.oppstrategy = [0,0,0]


    #  we need to define our training method
    def adjustStrategy(self, regrets):
        # the normalizing sum is intended to make sure that all the regrets sum up to 1
        normalizing_sum = 0

        regrets_clipped = np.clip(regrets, a_min=0, a_max=None)

        # initialize the normalizing sum to be the sum of all the regrets
        for i in range(self.num_actions):
            normalizing_sum += regrets_clipped[i]
        
        # if the sum of all the regrets is not 0
        if normalizing_sum > 0:
            # normalize the regrets - make sure that the probability sums to 1
            regrets_clipped = regrets_clipped / normalizing_sum
        else:
            # set the probabilities of each action to 0.33333 (equal)
            equal = 1 / self.num_actions
            regrets_clipped = [equal,equal,equal]
        
        # return the sum of all the regrets
        return np.sum(regrets_clipped)
    
    # choose an action according to the frequency of actions
    def chooseAction(self, strategy):
        # choose from the actions following the strategy
        return choice(self.actions, strategy)
    

    # compute the average strategy over time
    def computeStrategyOverTime(self, strategySum):
        strategy = [0,0,0]

        # the normalizing sum is intended to make sure that all the regrets sum up to 1
        normalizing_sum = 0

        # initialize the normalizing sum to be the sum of all the elements in the provided array
        for i in range(self.num_actions):
            normalizing_sum += strategySum[i]
        
        # for each action
        for i in range(self.num_actions):
            # check the normalizing sum
            if normalizing_sum > 0:
                strategy[i] = strategySum
            else:
                # equal preference to all actions
                strategy[i] = 1 / self.num_actions
            
        return strategy

 
    # return the utility for player 1 for a pair of actions
    # action1 is player 1's action
    def utility(self, action1, action2):
        # draw if the same action
        if action1 == action2:
            return 0
        
        # -------- rock
        elif action1 == 0:
            # rock vs paper
            if action2 == 1:
                # lose
                return -1
            else:
                # rock vs. scissors
                return 1
        
        # paper
        elif action1 == 1:
            # -- vs rock
            if action2 == 0:
                # win paper vs rock
                return 1
            else:
                # lose paper vs scissors
                return -1
        # scissors
        else:
            # vs rock
            if action2 == 0:
                # lose
                return -1
            else:
                # vs paper win
                return 1
            
    def train(self, num_iterations):

        for i in range(num_iterations):
            # compute the strategy of our opponent and ourself
            our_strat = self.adjustStrategy(self.regrets)
            their_strat = self.adjustStrategy(self.oppregrets)

            # log the strategies
            self.strategy += our_strat
            self.oppstrategy += their_strat

            # get the action that we take and the action that our opp takes
            our_action = self.chooseAction(our_strat)
            their_action = self.chooseAction(their_strat)

            # compute the utilities that we get from following our strategy and their strategy
            our_utility = self.utility(our_action, their_action)
            their_utility = self.utility(their_action, our_action)

            # for every action
            for action in self.actions:
                # seeing what could have been
                our_regret = self.utility(action, their_action) - our_utility
                their_regret = self.utility(action, our_action) - their_utility
                self.regrets[action] += our_regret
                self.oppregrets[action] += their_regret
    
def main():
    trainer = rock_paper_scissors()
    trainer.train(10000)
    target_policy = trainer.computeStrategyOverTime(trainer.strategy)
    opp_target_policy = trainer.computeStrategyOverTime(trainer.oppstrategy)
    print('player 1 policy: %s' % target_policy)
    print('player 2 policy: %s' % opp_target_policy)

if __name__ == "__main__":
    main()