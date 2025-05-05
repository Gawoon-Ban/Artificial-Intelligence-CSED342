from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def minimax(state, depth, agentIndex):
        # 게임이 종료 or depth가 0인 경우 평가 함수 호출
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents # 팩맨 -> 유령1 -> 유령2 -> ... -> 팩맨
        nextDepth = depth if nextAgent != 0 else depth - 1 # 모든 agent가 한번씩 행동하면 depth 1 감소소
        
        legalActions = state.getLegalActions(agentIndex) #현재 에이전트가 할 수 있는 행동
        if not legalActions: # 가능한 행동이 없으면 끝끝
            return self.evaluationFunction(state), None
            
        # 팩맨의 경우
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = minimax(successor, nextDepth, nextAgent) # 재귀적으로 값 계산
                if value > bestValue: # 더 높은 value 찾으면 갱신
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
        
        # 유령의 경우
        else:  # Ghosts (min nodes)
            bestValue = float('inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = minimax(successor, nextDepth, nextAgent)
                if value < bestValue: # 유령은 min value를 찾아야함함
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
    
    # 함수를 루트 노드에서 실행하여 최종 답변을 생성성
    _, action = minimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def minimax(state, depth, agentIndex):
        # 게임이 종료 or depth가 0인 경우 평가 함수 호출
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)
            
        # 팩맨의 경우    
        if agentIndex == 0:  # Pac-Man (max node)
            # 모든 succ에 대해 minimax 계산해서 최대값 반환
            return max(minimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
        else:  # Ghosts (min nodes)
            return min(minimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
    
    successor = gameState.generateSuccessor(0, action)
    return minimax(successor, self.depth, 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

### BEGIN_WRITTEN_SOLUTION
#Minimax는 유령이 팩맨의 점수를 최소화하려는 최악의 행동을 선택한다고 가정한다. 반면 Expectimax는 유령이 무작위 (같은 확률)로 행동한다고 가정한다
#팩맨이 유령에게 직접적으로 돌진하지 않는 이유는, 무작위 행동이지만 유령과의 충돌 위험이 높기 때문이다. 
### END_WRITTEN_SOLUTION

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state), None
            
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectimax(successor, nextDepth, nextAgent)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
        
        # 여기가 다르다
        else:  # Ghosts (expectation nodes)
            totalValue = 0
            # 균등 확률로 유령 이동
            prob = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectimax(successor, nextDepth, nextAgent)
                totalValue += prob * value #***********************************
            return totalValue, None
    
    _, action = expectimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)
            
        if agentIndex == 0:  # Pac-Man (max node)
            return max(expectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
        else:  # Ghosts (expectation nodes)
            prob = 1.0 / len(legalActions)
            #*************************  
            # 균등 확률로 유령 이동
            #*************************
            return sum(prob * expectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
    
    successor = gameState.generateSuccessor(0, action)
    return expectimax(successor, self.depth, 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

### BEGIN_WRITTEN_SOLUTION
# biased-expectimax에서는 유령이 정지할 확률이 높다. 패배 상황 (유령이 팩맨을 양쪽에서 가두는 위치)에서는 유령의 정지 확률이 높아도 결국 접근하며 팩맨을 잡는다.
# 이때 팩맨이 움직임을 멈추는 이유는 모든 legal 행동이 유령과의 충돌로 이어지는 상황에서, 더 이상 유효한 이동이 없기 때문이다. 
### END_WRITTEN_SOLUTION

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def biasedExpectimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state), None
            
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = biasedExpectimax(successor, nextDepth, nextAgent)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
        else:  # Ghosts (biased expectation nodes)
            totalValue = 0
            numActions = len(legalActions)
            # 정지할 확률이 다른 행동보다 높다
            stopProb = 0.5 + 0.5 / numActions
            otherProb = 0.5 / numActions
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = biasedExpectimax(successor, nextDepth, nextAgent)
                # **************************
                # 정지할 확률이 다른 행동보다 높다
                # **************************
                prob = stopProb if action == Directions.STOP else otherProb 
                totalValue += prob * value
            return totalValue, None
    
    _, action = biasedExpectimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def biasedExpectimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)
            
        if agentIndex == 0:  # Pac-Man (max node)
            return max(biasedExpectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
        else:  # Ghosts (biased expectation nodes)
            totalValue = 0
            numActions = len(legalActions)
            # 정지할 확률이 다른 행동보다 높다
            stopProb = 0.5 + 0.5 / numActions 
            otherProb = 0.5 / numActions
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = biasedExpectimax(successor, nextDepth, nextAgent)
                # 정지는 따로 고려하여 확률 적기
                prob = stopProb if action == Directions.STOP else otherProb
                totalValue += prob * value
            return totalValue
    
    successor = gameState.generateSuccessor(0, action)
    return biasedExpectimax(successor, self.depth, 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing SoftmaxExpectimaxAgent

### BEGIN_WRITTEN_SOLUTION
#1. alpha가 0일 때, 유령이 균등 확률로 무작위 행동하기에 Expectimax와 동일하다. alpha = 1.0인 상황에서는 유령이 낮은 점수 상태를 약간 선호하고, 약간 공격적으로 변한다.
# alpha가 10.0일때는 유령이 팩맨 점수를 최소화하려는 행동을 강하게 선호하기에, Minimax와 유사하게 공격적이다. 팩맨은 alpha가 증가할수록 유령의 공격성을 피해 조심스러운 전략을 택할 것이다.
#2. alpha가 0일 때, 유령이 무작위로 움직여 평균 점수 450을 얻었다. alpha = 1.0에서는 유령이 더 공격적으로 움직이고 점수가 400으로 약간 감소했다.
# alpha = 10.0일 때, 유령이 팩맨을 적극적으로 추적하여 점수가 345.9로 감소했다.
### END_WRITTEN_SOLUTION

class SoftmaxExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your SoftmaxExpectimaxAgent agent (problem 4)
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2', alpha='1.0'):
        super().__init__(evalFn, depth)
        self.alpha = float(alpha)

    def getAction(self, gameState):
        """
        Returns the softmax-expectimax action using self.depth and self.evaluationFunction.

        Ghosts (agentIndex >= 1) should be modeled as choosing their actions
        according to a softmax distribution:
            p(a) ∝ exp(-alpha * Value(successorState))
        where alpha is self.alpha.
        Pac-Man (agentIndex = 0) should still act like a max node.
        """

        # BEGIN_YOUR_ANSWER
        def softmaxExpectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth if nextAgent != 0 else depth - 1
            
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None
                
            if agentIndex == 0:  # Pac-Man (max node)
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = softmaxExpectimax(successor, nextDepth, nextAgent)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:  # Ghosts (softmax expectation nodes)
                values = []
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = softmaxExpectimax(successor, nextDepth, nextAgent)
                    values.append(value)
                # Compute weights inversely proportional to shifted values
                min_value = min(values)
                weights = [1.0 / (1.0 + self.alpha * (v - min_value)) if self.alpha != 0 else 1.0
                          for v in values]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(values)] * len(values)
                totalValue = sum(p * v for p, v in zip(probs, values))
                return totalValue, None
        
        _, action = softmaxExpectimax(gameState, self.depth, 0)
        return action
        # END_YOUR_ANSWER

    def getQ(self, gameState, action):
        """
        Returns the softmax-expectimax Q-Value using self.depth and self.evaluationFunction.
        """

        # BEGIN_YOUR_ANSWER
        def softmaxExpectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth if nextAgent != 0 else depth - 1
            
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
                
            if agentIndex == 0:  # Pac-Man (max node)
                return max(softmaxExpectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                          for a in legalActions)
            else:  # Ghosts (softmax expectation nodes)
                values = [softmaxExpectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                          for a in legalActions]
                # Compute weights inversely proportional to shifted values
                min_value = min(values)
                weights = [1.0 / (1.0 + self.alpha * (v - min_value)) if self.alpha != 0 else 1.0
                          for v in values]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(values)] * len(values)
                return sum(p * v for p, v in zip(probs, values))
        
        successor = gameState.generateSuccessor(0, action)
        return softmaxExpectimax(successor, self.depth, 1)
        # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing expectiminimax

### BEGIN_WRITTEN_SOLUTION
# Minimax는 유령이 팩맨의 점수를 최소하하려한다고 가정한다. Expectiminimax는 홀수 유령은 min 노드, 짝수 유령은 무작위 expectation 노드로 행동한다. depth 1,2,3에서는 게임 트리가 짧아서 짝수 유령의 무작위 행동이
# 결과에 큰 영향을 미치지 않기에 값이 동일하다. 그러나 depth 4부터는 짝수 유령의 무작위성이 누적되어서 일부 유령의 우호적인 행동이 포함된다.
# 이로인해 Expectiminimax의 Q value가 Minimax보다 높아진다.
### END_WRITTEN_SOLUTION

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectiminimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state), None
            
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectiminimax(successor, nextDepth, nextAgent)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
        elif agentIndex % 2 == 1:  # Odd-numbered ghosts (min nodes)
            bestValue = float('inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectiminimax(successor, nextDepth, nextAgent)
                if value < bestValue:
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction
        else:  # Even-numbered ghosts (expectation nodes)
            totalValue = 0
            prob = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectiminimax(successor, nextDepth, nextAgent)
                totalValue += prob * value
            return totalValue, None
    
    _, action = expectiminimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectiminimax(state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)
            
        if agentIndex == 0:  # Pac-Man (max node)
            return max(expectiminimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
        elif agentIndex % 2 == 1:  # Odd-numbered ghosts (min nodes)
            return min(expectiminimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
        else:  # Even-numbered ghosts (expectation nodes)
            prob = 1.0 / len(legalActions)
            return sum(prob * expectiminimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                      for a in legalActions)
    
    successor = gameState.generateSuccessor(0, action)
    return expectiminimax(successor, self.depth, 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 6)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def alphaBeta(state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state), None
            
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                # 알파베타 가지치기
                alpha = max(alpha, bestValue)
                if bestValue > beta:
                    return bestValue, bestAction
            return bestValue, bestAction
        elif agentIndex % 2 == 1:  # Odd-numbered ghosts (min nodes)
            bestValue = float('inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                if value < bestValue:
                    bestValue = value
                    bestAction = action
                beta = min(beta, bestValue)
                if bestValue < alpha:
                    return bestValue, bestAction
            return bestValue, bestAction
        else:  # Even-numbered ghosts (expectation nodes)
            totalValue = 0
            prob = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                totalValue += prob * value
            return totalValue, None
    
    _, action = alphaBeta(gameState, self.depth, 0, float('-inf'), float('inf'))
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def alphaBeta(state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth - 1
        
        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)
            
        if agentIndex == 0:  # Pac-Man (max node)
            bestValue = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                if bestValue > beta:
                    return bestValue
            return bestValue
        elif agentIndex % 2 == 1:  # Odd-numbered ghosts (min nodes)
            bestValue = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                if bestValue < alpha:
                    return bestValue
            return bestValue
        else:  # Even-numbered ghosts (expectation nodes)
            prob = 1.0 / len(legalActions)
            totalValue = 0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                totalValue += prob * value
            return totalValue
    
    successor = gameState.generateSuccessor(0, action)
    return alphaBeta(successor, self.depth, 1, float('-inf'), float('inf'))
    # END_YOUR_ANSWER

######################################################################################
# Problem 7a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 7).
  """

  # BEGIN_YOUR_ANSWER
  pacmanPos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  score = currentGameState.getScore()
  
  # Food distance feature
  foodList = food.asList()
  if foodList:
      foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
      minFoodDist = min(foodDistances)
      foodScore = 1.0 / (minFoodDist + 1)  # Reciprocal to prioritize closer food
  else:
      foodScore = 1000  # Large reward for eating all food
  
  # Ghost distance feature
  ghostScore = 0
  for ghost in ghostStates:
      dist = manhattanDistance(pacmanPos, ghost.getPosition())
      if ghost.scaredTimer > 0:
          if dist < ghost.scaredTimer:
              ghostScore += 200 / (dist + 1)  # Reward for chasing scared ghosts
      else:
          if dist < 1.5:
              ghostScore -= 1500 / (dist + 1)  # Strong penalty for very close active ghosts
          elif dist < 2:
              ghostScore -= 750 / (dist + 1)  # Moderate penalty for close active ghosts
          else:
              ghostScore += 10 / (dist + 1)  # Small reward for staying away
  
  # Capsule feature
  capsules = currentGameState.getCapsules()
  if capsules:
      capsuleDistances = [manhattanDistance(pacmanPos, cap) for cap in capsules]
      minCapsuleDist = min(capsuleDistances)
      capsuleScore = 100 / (minCapsuleDist + 1)  # Encourage moving toward capsules
  else:
      capsuleScore = 0
  
  # Food count feature
  foodCountScore = 500 / (len(foodList) + 1) if foodList else 0
  
  # Combine features
  return score + 3 * foodScore + ghostScore + capsuleScore + foodCountScore
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 7.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction