import random

class QLearner:
    def __init__(self, actions, alpha=0.01, gamma=0.01, epsilon=0.05):
        # Learning Rate
        self.alpha = alpha
        # Discount Factor
        self.gamma = gamma
        # Exploration probability
        self.epsilon = epsilon

        self.actions = actions

        self.q = {}

    def getQ(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def getTotalReward(self):
        return sum([self.q.get(key) for key in self.q.keys()])

    def updateQ(self, state, action, reward, state2):
        currentQ = self.getQ(state, action)

        if currentQ is 0:
            self.q[(tuple(state), action)] = reward
        else:
            futureQ = max([self.getQ(state2, futureAction) for futureAction in self.actions])
            newQ = currentQ + self.alpha * (reward + futureQ - self.gamma * currentQ)

            self.q[(tuple(state), action)] = newQ

    def updateActions(self, newActions):
        self.actions = newActions

    def chooseAction(self, state):
        # Explore
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Exploit
        else:
            actionRewards = [self.getQ(state, action) for action in self.actions]
            maxActionReward = max(actionRewards)

            # We could have more than one action with max reward; choose one randomly.
            count = actionRewards.count(actionRewards)

            if count > 1:
                best = [i for i in range(len(self.actions)) if actionRewards[i] == maxQ]
                i = random.choice(best)
            else:
                i = actionRewards.index(maxActionReward)

            return self.actions[i]
