# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodlist = newFood.asList()
        proportion = 1.0
        #Score Evaluation
        score = 0
        #Is the new state a good state that it gives us a positive score?
        scoreDiff = successorGameState.getScore() - currentGameState.getScore()
        if scoreDiff > 0:
            score += 10
            return score
        # Distance of closest food.
        distFood = 0
        foods = []
        for food in foodlist:
            foods.append((manhattanDistance(food, newPos)))
        #Reciprocal Evaluation
        score += proportion/min(foods)

        # Distance of closest ghost.
        distGhost = 0
        closestGhost = None
        for ghost in newGhostStates:
            tmpDist = manhattanDistance(newPos, ghost.getPosition())
            if (distGhost == 0) or (distGhost > tmpDist):
                distGhost = tmpDist
                closestGhost = ghost
        #Landed on a ghost, bad!
        if distGhost == 0:
            return -99
        #Not good to stop in pacman, keep moving.
        if action == 'Stop':
            return -99

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, agentIndex): 
            if gameState.isLose() or gameState.isWin() or depth == 0: #check not terminal state
                return self.evaluationFunction(gameState) 
            value = -9999999999 #-infinte
            for action in gameState.getLegalActions(0): #legal moves
                value = max(value, minValue(gameState.generateSuccessor(0, action), depth, 1)) # max(value, min(successor state, depth))
            return value
 
        def minValue(gameState, depth, agentIndex):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            value = 9999999999 #infinte
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1))
                else:
                    value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return value
        
        bestScore = -9999999999
        bestAction = None

        for action in gameState.getLegalActions():
            score = max(bestScore, minValue( gameState.generateSuccessor(0, action), self.depth, 1))
            if score > bestScore:
                bestScore = score
                bestAction = action
 
        return bestAction
        #dies at nearest ghost because it assumes optimal adversary
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, agentIndex, alpha, beta): #add parameter alpha and beta.
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            value = -9999999999
            for action in gameState.getLegalActions(0):
                value = max(value, minValue(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
                if value > beta:    #this is the pruning part
                    return value
                alpha = max(alpha, value)
            return value
 
        def minValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            value = 9999999999
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1, alpha, beta))
                else:
                    value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
    
        bestScore = -9999999999
        bestAction = None
        alpha = -9999999999
        beta = 9999999999
        for action in gameState.getLegalActions():
            score = max(bestScore, minValue(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta))
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(score, alpha)
 
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, agentIndex):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            value = -9999999999
            for action in gameState.getLegalActions(0):
                value = max(value, exp(gameState.generateSuccessor(0, action), depth, 1))
            return value
 
        def exp(gameState, depth, agentIndex):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            value = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    value = value + maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1)
                else:
                    value = value + exp(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            return value/len(gameState.getLegalActions(agentIndex))  #value/all legal moves

        bestScore = -9999999999
        bestAction = None
        for action in gameState.getLegalActions():
            score = max(bestScore, exp(gameState.generateSuccessor(0, action), self.depth, 1))
            if score > bestScore:
                bestScore = score
                bestAction = action
 
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function finds the closest food and uses the reciprocal function
      to calculate a good evaluation from its current state. The "proportion" reciprocal allows us to calculate an evaluation.
      For example, if the closest food's distance is low, then the foodEval will be higher (i.e. dividing by a lower number = higher results).
      The same applies for the ghostEval, which accounts for the closest ghost. If it is a scared ghost, to favor the evaluation, and if it is not
      a scared ghost, to not favor the evaluation. It tells us to pick the action to chase the ghost if it is scared and to run if it is not scared.
      The farther away a not-scared ghost is, the better the evaluation. The closer away a not-scared ghost is, the worst the evaluation.
    """
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    "*** YOUR CODE HERE ***"
    foodlist = Food.asList()
    distFood = 0
    distGhost = 0
    closestGhost = None
    ghostEval = 0
    foodEval = 0
    proportion = 1.0

    # Distance of closest food.
    for food in foodlist:
      tmpDist = manhattanDistance(food, Pos)
      if (distFood == 0) or (distFood > tmpDist):
        distFood = tmpDist
    if distFood != 0:
      foodEval += proportion/distFood
    # Distance of closest ghost.
    for ghost in GhostStates:
        tmpDist = manhattanDistance(Pos, ghost.getPosition())
        if (distGhost == 0) or (distGhost > tmpDist):
            distGhost = tmpDist
            closestGhost = ghost
    
    if distGhost > 0:
        if closestGhost.scaredTimer > 0:
            ghostEval += proportion/distGhost
        else:
            ghostEval -= proportion/distGhost

    # return ghostEval + foodEval + otherEval
    return currentGameState.getScore() + ghostEval + foodEval

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

