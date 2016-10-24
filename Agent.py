import itertools

from QLearner import QLearner

class BaseAgent:
    def __init__(self, stateTemplate=(), statesMap={}, actions=[], rewardMatrix=[], initialState=None):
        self._currentState = initialState

        self._stateTemplate = stateTemplate

        # statesMap ={
        #     state1: ['value1', 'value2']
        #     state2: ['value1', 'value2', 'value3']
        #     ...
        # }
        self._statesMap = statesMap

        self._actions = actions

        self.qLearner = QLearner(
            states=self._generateStateSpace(self._statesMap, self._stateTemplate), \
            actions=self._actions, \
            rewardMatrix=rewardMatrix \
        )

    def _generateStateSpace(self, statesMap, stateTemplate):
        stateValuesList = [statesMap[state] for state in stateTemplate]
        return itertools.product(*stateValuesList)

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

        BaseAgent.__init__(stateTemplate=self._stateTemplate, statesMap=self.statesMap, \
            actions=self.actions, rewardMatrix=self.rewardMatrix, initialState=self.initialState)
