import numpy as np
from numpy.random import choice

# A class that implements rock paper scissors as a CFR bot
class RockPaperScissorsAgent:
    # initialize all the variables
    def __init__(self):

        # there are 3 possible actions
        self.num_actions = 3
        # corresponding to rock, paper, scissors
        self.possible_actions = [0,1,2]

        # variables to store our policy and regrets
        self.regrets = [0.0,0.0,0.0]
        self.policy = [0.0,0.0,0.0]

        # we must keep track of our opponents policy and regrets to make sure we can adjust our policy
        self.opponent_regrets = [0.0,0.0,0.0]
        self.opponent_policy = [0.0,0.0,0.0]

    # This method takes in the accumulated sum of regrets and returns a frequency of each of the 
    # three actions that our bot / player should play.
    # This method only computes the policy for the next iteration.
    def get_policy(self, regret_sum):
        # make sure that the regrets can never be negative
        regrets_clipped = np.clip(regret_sum, a_min=0, a_max=None)

        # the normalizing sum is intended to make sure that all the regrets sum up to 1
        normalizing_sum = 0

        # initialize the normalizing sum to be the sum of all the elements in clipped regrets
        for i in range(self.num_actions):
            normalizing_sum += regrets_clipped[i]

        # If there is at least 1 positive regret
        if normalizing_sum > 0:
            # normalize all the regrets 
            regrets_clipped = regrets_clipped / normalizing_sum
        
        # if all the regrets are 0, this means that no action is better than the others
        else:
            # set the regrets to be equal for everything - as all actions are equally good as of now
            # this sets the regrets_clipped array to [1/3 ,  1/3 , 1/3]
            regrets_clipped = np.repeat(1/self.num_actions, self.num_actions)
        
        # we return our array of regrets
        return regrets_clipped

    # compute the average policy over time
    # This method computes the average policy over all the iterations.
    # The policy so far is the policy that our bots have accumulated over the iterations
    def get_policy_over_time(self, policy_so_far):
        updated_average_policy = [0, 0, 0]

        # the normalizing sum is intended to make sure that all the regrets sum up to 1
        normalizing_sum = 0

        # initialize the normalizing sum to be the sum of all the elements in the policy so far
        for i in range(self.num_actions):
            normalizing_sum += policy_so_far[i]

        # for every action in the policy
        for i in range(self.num_actions):
            # if we have computed at least one positive regret - seen one action as better than another
            if normalizing_sum > 0:
                # normalize the average policy values
                updated_average_policy[i] = policy_so_far[i] / normalizing_sum
            
            # else - we have found all actions to be equally good - regret = 0
            else:
                # the policy we should then follow is an equal preference for every action
                updated_average_policy[i] = 1 / self.num_actions
        
        # return the average policy that we have computed
        return updated_average_policy

    # choose an action from the list of actions at a frequency consistent with the provided policy
    def get_action(self, policy):
        # return the action
        return choice(self.possible_actions, p=policy)


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

    # the method to train our agent
    def train(self, iterations):

        # train the bot iterations times
        for _ in range(iterations):
            # get our policy - stored in the fields
            hero_policy = self.get_policy(self.regrets)
            # add it to our current policy to accumulate strategies over time
            self.policy += hero_policy

            # get our opponents policy and do the same 
            villain_policy = self.get_policy(self.opponent_regrets)
            self.opponent_policy += villain_policy

            # compute the action that we play as well as the action that our opponent plays
            hero_action = self.get_action(hero_policy)
            villain_action = self.get_action(villain_policy)

            # log the actions by obtaining their utility for the respective players
            hero_reward = self.utility(hero_action, villain_action)
            villain_reward = self.utility(villain_action, hero_action)

            # for every possible action
            for action in range(self.num_actions):
                # calculate the hero's regret that stems from not taking that action
                hero_regret = self.utility(action, villain_action) - hero_reward
                # and add the regret to our running sum
                self.regrets[action] += hero_regret 

                # do the same for the villain to track their policy
                opp_regret = self.utility(action, hero_action) - villain_reward
                self.opponent_regrets[action] += opp_regret


def main():
    # create an agent
    agent = RockPaperScissorsAgent()
    # train it for 10,000 iterations
    agent.train(10000)

    # obtain the final policies after training
    target_policy = agent.get_policy_over_time(agent.policy)
    opp_target_policy = agent.get_policy_over_time(agent.opponent_policy)
    # print the policies
    print('player 1 policy: %s' % target_policy)
    print('player 2 policy: %s' % opp_target_policy)

# for running purposes
if __name__ == "__main__":
    main()