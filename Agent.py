import itertools
import nump as np

from QLearner import QLearner

class BaseAgent:
    def __init__(self, states=[], actions=[], rewardMatrix=[], initialState=None):
        self._currentState = initialState

        self.qLearner = QLearner(states=states, actions=actions, rewardMatrix=rewardMatrix)

    def getCurrentState(self):
        return self._currentState

    def _setCurrentState(self, newState):
        self._currentState = newState

    def performNextAction(self):
        currentState = self.getCurrentState()
        actionToPerform = self.qLearner.chooseAction(currentState)

        # Perform Action
        # TODO: Implement this
        nextState = None
        reward = rewardMatrix[currentState, nextState]

        self._setCurrentState(nextState)
        self.qLearner.updateQ(state, action, reward, nextState)

        return reward, nextState

class InstructionGenerationAgent(BaseAgent):
    def __init__(self):
        self._stateTemplate = ('task-trajectory', 'target-in-view', 'is-near-landmark', 'is-confused', 'prev-instruction-length')

        self.statesMap = {
            'task-trajectory': ['closer', 'further', 'no-change', 'finished'],
            'target-in-view': ['yes', 'no'],
            'is-near-landmark': ['yes', 'no'],
            'is-confused': ['yes', 'no'],
            'prev-instruction-length': ['short', 'medium', 'long', 'none']
        }

        self.actions = ['plain', 'turn', 'plain-landmark', 'turn-landmark']

        self.initialState = ('no-change', 'no', 'no', 'no', 'none')
        self.states = self.generateStateSpace(self.statesMap, self.stateTemplate)

        # REWARD MATRIX
        # -10 for every move
        # +100 for finishing
        # +1 for getting closer
        self.rewardMatrix = np.array([[-10 for state in self._states] for state in self._states])

        for state in self.states:
            for finishState in filter(lambda state: state[0] is 'finished', self.states):
                self.rewardMatrix[state, finishState] = 100

        for state in self.states:
            for finishState in filter(lambda state: state[0] is 'closer', self.states):
                self.rewardMatrix[state, finishState] = 1

        BaseAgent.__init__(states=self.states, \
            actions=self.actions, rewardMatrix=self.rewardMatrix, initialState=self.initialState)

        def generateStateSpace(self, statesMap, stateTemplate):
            stateValuesList = [statesMap[state] for state in stateTemplate]
            return itertools.product(*stateValuesList)

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
