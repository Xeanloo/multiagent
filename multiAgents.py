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
import random, util

from game import Agent
bestAction = None

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        score = 0
        walls = currentGameState.getWalls().asList()
        newFood = successorGameState.getFood().asList()
        oldfoodPos = currentGameState.getFood().asList()
        ghostPos = successorGameState.getGhostPositions()
        foodFound = False
        dist = 0

        if (newPos[0] + 1, newPos[1]) in newFood:
            score += 2
            foodFound = True
        if (newPos[0] - 1, newPos[1]) in newFood:
            score += 2
            foodFound = True
        if (newPos[0], newPos[1] + 1) in newFood:
            score += 2
            foodFound = True
        if (newPos[0], newPos[1] - 1) in newFood:
            score += 2
            foodFound = True

        if not foodFound:
            if newPos in oldfoodPos:
                score += 10
            if newFood:
                dist = manhattanDistance(newPos, newFood[0])
                for food in newFood:
                    if manhattanDistance(food, newPos) < dist:
                        dist = manhattanDistance(food, newPos)
            score -= dist/2

        for ghost in ghostPos:
            if manhattanDistance(ghost, newPos) < 2:
                score -= 10

        return successorGameState.getScore() + score

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

    def miniMax(self, agent, depth, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0:
            return max(self.miniMax(1, depth, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))
        else:
            if agent == gameState.getNumAgents()-1:
                return min(self.miniMax(0, depth+1, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))
            else:
                return min(self.miniMax(agent+1, depth, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.miniMax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    global bestAction
    bestAction = None

    def alphaBeta(self, agent, depth, gameState, alpha, beta):
        global bestAction

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = -float("inf")
            for action in gameState.getLegalActions(agent):
                tempValue = self.alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta)
                if tempValue > value:
                    value = tempValue
                    bestAction = action if depth == 0 else bestAction
                if value > beta:
                    return value
                alpha = max(alpha, value)
        else:
            value = float("inf")
            if agent == gameState.getNumAgents()-1:
                newAgent = 0
                newDepth = depth+1
            else:
                newAgent = agent+1
                newDepth = depth

            for action in gameState.getLegalActions(agent):
                value = min(value, self.alphaBeta(newAgent, newDepth, gameState.generateSuccessor(agent, action), alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)

        return value

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        global bestAction
        global bestScore
        alpha = -float('inf')
        beta = float('inf')

        self.alphaBeta(0, 0, gameState, alpha, beta)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    global bestAction
    bestScore = None

    def expectimax(self, agent, depth, gameState):
        global bestAction

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = -float("inf")
            for action in gameState.getLegalActions(agent):
                tempValue = self.expectimax(1, depth, gameState.generateSuccessor(agent, action))
                if tempValue > value:
                    value = tempValue
                    bestAction = action if depth == 0 else bestAction
            return value
        else:
            possibleActions = gameState.getLegalActions(agent)
            if agent == gameState.getNumAgents() - 1:
                newAgent = 0
                newDepth = depth + 1
            else:
                newAgent = agent + 1
                newDepth = depth
            value = sum(self.expectimax(newAgent, newDepth, gameState.generateSuccessor(agent, action)) for action in possibleActions)
            return value/len(possibleActions)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.expectimax(0, 0, gameState)
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newCaps = currentGameState.getCapsules()
    capsCount = len(newCaps)
    foodCount = len(newFood)

    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newScore = currentGameState.getScore()

    #if an unscared ghost ist too close, run away asap
    ghostIndex = 0
    for ghostPos in newGhostPos:
        if manhattanDistance(newPos, ghostPos) < 2 and newScaredTimes[ghostIndex] == 0:
            return -999999
        ghostIndex += 1

    # Calculate the distance to the closest food, if there's any left
    minFoodDist = 999999
    if foodCount > 0:
        for food in newFood:
            newDist = manhattanDistance(newPos, food)
            if newDist < minFoodDist:
                minFoodDist = newDist
    else:
        minFoodDist = 0.1

    # Calculate the distance to the closest capsule, if there's any left
    minCapsuleDist = 999999
    if capsCount > 0:
        for capsule in newCaps:
            newDist = manhattanDistance(newPos, capsule)
            if newDist < minCapsuleDist:
                minCapsuleDist = newDist
    else:
        minCapsuleDist = 0.1

    # return sum of all parameters multiplied with their respective weights
    return newScore + 10.0/minFoodDist - 10.0/minCapsuleDist - capsCount*2 - foodCount*100


# Abbreviation
better = betterEvaluationFunction
