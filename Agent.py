import itertools
import numpy as np

from RLQLearner import QLearner

class BaseAgent(object):
    def __init__(self, states=[], actions=[], rewardMatrix=[], initialState=None):
        self._currentState = initialState

        self.qLearner = QLearner(states=states, actions=actions, rewardMatrix=rewardMatrix)

    def getCurrentState(self):
        return self._currentState

    def _setCurrentState(self, newState):
        self._currentState = newState

    def getNextAction(self):
        currentState = self.getCurrentState()
        actionToPerform = self.qLearner.chooseAction(currentState)

        return actionToPerform

    def updateState(self, currentState, actionToPerform, nextState):
        reward = self.qLearner.getReward(currentState, nextState)
        self._setCurrentState(nextState)
        self.qLearner.updateQ(currentState, actionToPerform, reward, nextState)

    def getRewardMatrix(self):
        return self.qLearner.getRewardMatrix()

    def getQMatrix(self):
        return self.qLearner.getQMatrix()

class InstructionGenerationAgent(BaseAgent):
    def __init__(self):
        self._stateTemplate = ['status']

        self.statesMap = {
            'status': ['done', 'not-done']
        }

        self.actions = ['affirm', 'bye', 'canthear', 'confirm-domain', 'negate', 'repeat',
            'reqmore', 'welcomemsg', 'canthelp', 'canthelp.missing_slot_value',
            'canthelp.exception', 'expl-conf', 'impl-conf', 'inform', 'offer', 'request', 'select',
            'welcomemsg']

        self.initialState = ('not-done')
        self.states = self.statesMap['status']

        # REWARD MATRIX
        # -10 for every move
        # +100 for finishing
        # +1 for getting closer
        self.rewardMatrix = np.array([[-10 for state in self.states] for state in self.states])

        self.rewardMatrix[0, range(len(self.states))] = 100

        super(InstructionGenerationAgent, self).__init__(states=self.states, actions=self.actions, rewardMatrix=self.rewardMatrix, initialState=self.initialState)

    def generateStateSpace(self, statesMap, stateTemplate):
        stateValuesList = [statesMap[state] for state in stateTemplate]
        return [s for s in stateValuesList[0]]

class InterventionAgent(BaseAgent):
    def __init__(self):
        self._stateTemplate = ('current-follower-action', 'task-trajectory', 'last-action')

        self.statesMap = {
            'current-follower-action': ['working', 'clarifiying', 'no-action'],
            'task-trajectory': ['closer', 'further', 'no-change', 'finished'],
            'last-action': ['instruction-given', 'instruction-clarified', 'instruction-followed', 'no-action']
        }

        self.actions = ['intervene', 'no-intervene']

        self.initialState = ('no-action', 'no-change', 'no-action')
        self.states = self.generateStateSpace(self.statesMap, self.stateTemplate)

        # REWARD MATRIX
        # -1 for every move
        # +1 for getting closer
        # +100 for finishing
        self.rewardMatrix = np.array([[-1 for state in self._states] for state in self._states])

        for state in self.states:
            for finishState in filter(lambda state: state[1] is 'finished', self.states):
                self.rewardMatrix[state, finishState] = 100

        for state in self.states:
            for finishState in filter(lambda state: state[1] is 'closer', self.states):
                self.rewardMatrix[state, finishState] = 1

        BaseAgent.__init__(states=self.states, \
            actions=self.actions, rewardMatrix=self.rewardMatrix, initialState=self.initialState)

    def generateStateSpace(self, statesMap, stateTemplate):
        stateValuesList = [statesMap[state] for state in stateTemplate]
        return itertools.product(*stateValuesList)

if __name__ == '__main__':
    agent = InstructionGenerationAgent()
    print agent.getRewardMatrix()
    print agent.getQMatrix()

    agent.updateState('not-done', 'inform', 'done')

    print agent.getRewardMatrix()
    print agent.getQMatrix()
