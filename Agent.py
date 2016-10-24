import itertools
import nump as np

from QLearner import QLearner

class BaseAgent:
    def __init__(self, states=[], actions=[], rewardMatrix=[], initialState=None):
        self._currentState = initialState

        # statesMap ={
        #     state1: ['value1', 'value2']
        #     state2: ['value1', 'value2', 'value3']
        #     ...
        # }

        self._states = states
        self._actions = actions

        self.qLearner = QLearner(
            states=self._states, \
            actions=self._actions, \
            rewardMatrix=rewardMatrix \
        )

    def getCurrentState(self):
        return self._currentState

    def _setCurrentState(self, newState):
        self._currentState = newState

    def performNextAction(self, ):
        currentState = self.getCurrentState()
        actionToPerform = self.qLearner.chooseAction(currentState)

        # Perform Action
        # --- Do something ---
        nextState = (0, 1, 0)
        reward = 0

        self._setCurrentState(nextState)
        self.qLearner.updateQ(state, action, reward, nextState)

        return reward, nextState

class InstructionGenerationAgent(BaseAgent):
    def __init__(self):
        self._stateTemplate = ()
        self.statesMap = {}
        self.actions = []
        self.rewardMatrix = []
        self.initialState = ()

        BaseAgent.__init__(states=self.generateStateSpace(self.statesMap, self.stateTemplate), \
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

        self.rewardMatrix = np.array([[0 for state in self._states] for state in self._states])

        for state in self.states:
            for finishState in itertools.product(statesMap['current-follower-action'], ['finished'], statesMap['last-action']):
                self.rewardMatrix[state, finishState] = 100

        BaseAgent.__init__(states=self.states, \
            actions=self.actions, rewardMatrix=self.rewardMatrix, initialState=self.initialState)

    def generateStateSpace(self, statesMap, stateTemplate):
        stateValuesList = [statesMap[state] for state in stateTemplate]
        return itertools.product(*stateValuesList)
