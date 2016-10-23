import random
import numpy as np

class QLearner:
    def __init__(self, states=[], actions=[], rewardMatrix=[], alpha=0.1, gamma=0.3, epsilon=0.15):
        # Learning Rate
        self._alpha = alpha
        # Discount Factor
        self._gamma = gamma
        # Exploration probability
        self._epsilon = epsilon

        self._states = states
        self._actions = actions

        # Reward Matrix:
        #
        #  S\A   | Action1  Action2  Action3
        # ---------------------------------
        # State1 | R11      R12      R13
        # State2 | R21      R22      R23
        # State3 | R31      R32      R33

        if not rewardMatrix:
            self._rewardMatrix = np.array([[0 for action in self._actions] for state in self._states])
        else:
            self._rewardMatrix = np.array(rewardMatrix)

        # Q-table initialized with same dimensions as Reward Matrix and all 0's.
        self._qMatrix = np.zero_like(self._rewardMatrix)

    def getReward(self, state, action):
        return self._rewardMatrix[self._states.index(state), self._actions.index(action)]

    def setReward(self, state, action, reward):
        self._rewardMatrix[self._states.index(state), self._actions.index(action)] = reward

    def getQ(self, state, action):
        return self._qMatrix[self._states.index(state), self._actions.index(action)]

    def setQ(self, state, action, q):
        self._qMatrix[self._states.index(state), self._actions.index(action)] = q

    def updateQ(self, state, action, reward, nextState):
        currentQ = self.getQ(state, action)
        futureMaxQ = max([self.getQ(nextState, futureAction) for futureAction in self._actions])

        # Sutton, R.S. and Barto, A.G. Reinforcement Learning: An Introduction, 1998.
        newQ = currentQ + self._alpha * (reward + self._gamma * futureMaxQ - currentQ)

        self.setQ(state, action, newQ)

    def chooseAction(self, state):
        # Explore
        if random.uniform(0, 1) < self._epsilon:
            return random.choice(self._actions)
        # Exploit
        else:
            actionRewards = [self.getQ(state, action) for action in self._actions]
            maxActionReward = max(actionRewards)

            # We could have more than one action with max reward; choose one randomly.
            count = actionRewards.count(actionRewards)

            if count > 1:
                best = [i for i in range(len(self._actions)) if actionRewards[i] == maxQ]
                i = random.choice(best)
            else:
                i = actionRewards.index(maxActionReward)

            return self._actions[i]
