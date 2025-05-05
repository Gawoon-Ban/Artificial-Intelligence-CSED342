#!/usr/bin/env python

import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

from game import Agent
from ghostAgents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os

import pacman, time, layout, textDisplay
textDisplay.SLEEP_TIME = 0
textDisplay.DRAW_EVERY = 1000
thismodule = sys.modules[__name__]

try:
    import solution
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False
grader.useSolution = solution_exist

def run(layname, pac, ghosts, nGames = 1, name = 'games', verbose=True):
  """
  Runs a few games and outputs their statistics.
  """
  if grader.fatalError:
    return {'time': 65536, 'wins': 0, 'games': None, 'scores': [0]*nGames, 'timeouts': nGames}

  starttime = time.time()
  lay = layout.getLayout(layname, 3)
  disp = textDisplay.NullGraphics()

  if verbose:
    print('*** Running %s on' % name, layname,'%d time(s).' % nGames)
  games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=False)
  if verbose:
    print('*** Finished running %s on' % name, layname,'after %d seconds.' % (time.time() - starttime))
  
  stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games], 'timeouts': [g.agentTimeout for g in games].count(True)}
  if verbose:
    print('*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games)))

  return stats

class RecordingReflexAgent(Agent):
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    # Save the state
    recordedStates.append(gameState)

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return successorGameState.getScore()

recordedStates = []
hiddenTestOpponents = 4
random.seed(SEED)
run('trickyClassic', RecordingReflexAgent(), [DirectionalGhost(i + 1) for i in range(hiddenTestOpponents)],
    name='recording', verbose=False)  # two ghosts


def testBasic(agentName):
  agent = {'minimax': submission.MinimaxAgent(depth=2),
           'expectimax': submission.ExpectimaxAgent(depth=2),
           'biased-expectimax': submission.BiasedExpectimaxAgent(depth=2),
           'expectiminimax': submission.ExpectiminimaxAgent(depth=2),
           'alphabeta': submission.AlphaBetaAgent(depth=2),
            'softmax': submission.SoftmaxExpectimaxAgent(evalFn='scoreEvaluationFunction', depth='2', alpha='1.0')
            }
  stats = run('smallClassic', agent[agentName], [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % (agentName, 2))
  if stats['timeouts'] > 0:
    grader.fail('Your ' + agentName + ' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
  grader.assignFullCredit()



gamePlay = {}
hiddenTestDepth = 2

def testHidden(agentFullName):
  player = 0
  depth = hiddenTestDepth
  subAgent = getattr(submission, agentFullName)(depth=depth)
  alpha_for_softmax = 1.0

  if solution_exist:
    solAgent = getattr(solution, agentFullName)(depth=depth)\

    num_states = 40

    for state in recordedStates[-num_states:]:
      pred = subAgent.getQ(state, subAgent.getAction(state))
      if solution_exist:
        answer = solAgent.getQ(state, solAgent.getAction(state))
        grader.requireIsEqual(answer, pred)  # compare values of successor states

    if agentFullName == 'AlphaBetaAgent':
      solExpectiminimaxAgent = solution.ExpectiminimaxAgent(depth=depth)

      def getQValues(agent):
        return [agent.getQ(state, agent.getAction(state))
                for state in recordedStates[-num_states:]]

      tm = graderUtil.TimeMeasure()
      tm.check()
      sol_qvalues = getQValues(solExpectiminimaxAgent)
      sol_time = tm.elapsed()

      tm.check()
      sub_qvalues = getQValues(subAgent)
      sub_time = tm.elapsed()

      print('ExpectiminimaxAgent: {} seconds'.format(sol_time))
      print('AlphaBetaAgent: {} seconds'.format(sub_time))

      grader.requireIsEqual(sol_qvalues, sub_qvalues)  # values of AlphaBetaAgent and ExpectiminimaxAgent should be same
      grader.requireIsLessThan(sol_time * 0.85, sub_time)  # AlphaBetaAgent should be faster than MinimaxAgent
    
    elif agentFullName == 'SoftmaxExpectimaxAgent':
      subAgent = getattr(submission, agentFullName)(depth=depth, alpha=str(alpha_for_softmax))
      if solution_exist:
          solAgent = getattr(solution, agentFullName)(depth=depth, alpha=str(alpha_for_softmax))


def experimentSoftmaxAlphaSeries(
    alpha_list=None, depth=2, nGames=5, layname='smallClassic'
):

    if alpha_list is None:
        alpha_list = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    for alpha in alpha_list:
        agent = submission.SoftmaxExpectimaxAgent(
            evalFn='scoreEvaluationFunction',
            depth=str(depth),
            alpha=str(alpha)
        )
        ghosts = [DirectionalGhost(i + 1) for i in range(2)]

        stats = run(
            layname,
            agent,
            ghosts,
            nGames=nGames,
            name=f'SoftmaxExpectimaxAgent(alpha={alpha})',
            verbose=False
        )
        win_count = stats['wins']
        avg_score = sum(stats['scores']) / len(stats['scores'])
        timeouts = stats['timeouts']

        results.append((alpha, win_count, avg_score, timeouts))

    print("==== SoftmaxExpectimaxAgent alpha-series experiment ====")
    print(f"Layout = {layname}, Depth = {depth}, nGames = {nGames}")
    for (alpha, wins, avgScore, timeouts) in results:
        print(f"alpha={alpha:<4}, wins={wins:2}, timeouts={timeouts:2}, avgScore={avgScore:7.2f}")
    print("=========================================================")

    return results

maxSeconds = 10

grader.addBasicPart('1a-1-basic', lambda : testBasic('minimax'), 1, maxSeconds=maxSeconds, description='Tests minimax for timeout on smallClassic.')
grader.addHiddenPart('1a-2-hidden', lambda : testHidden('MinimaxAgent'), 1, maxSeconds=maxSeconds, description='Tests minimax')

grader.addBasicPart('2a-1-basic', lambda : testBasic('expectimax'), 1, maxSeconds=maxSeconds, description='Tests expectimax for timeout on smallClassic.')
grader.addHiddenPart('2a-2-hidden', lambda : testHidden('ExpectimaxAgent'), 1, maxSeconds=maxSeconds, description='Tests expectimax')

grader.addBasicPart('3a-1-basic', lambda : testBasic('biased-expectimax'), 1, maxSeconds=maxSeconds, description='Tests biased-expectimax for timeout on smallClassic.')
grader.addHiddenPart('3a-2-hidden', lambda : testHidden('BiasedExpectimaxAgent'), 5, maxSeconds=maxSeconds, description='Tests biased-expectimax')

grader.addBasicPart('4a-1-basic',lambda: testBasic('softmax'), 1, maxSeconds=maxSeconds, description='Tests softmax-expectimax for timeout on smallClassic.')
grader.addHiddenPart('4a-2-hidden', lambda: testHidden('SoftmaxExpectimaxAgent'), 5, maxSeconds=maxSeconds, description='Tests softmax-expectimax')

grader.addBasicPart('5a-1-basic', lambda : testBasic('expectiminimax'), 1, maxSeconds=maxSeconds, description='Tests expectiminimax for timeout on smallClassic.')
grader.addHiddenPart('5a-2-hidden', lambda : testHidden('ExpectiminimaxAgent'), 3, maxSeconds=maxSeconds, description='Tests expectiminimax')

grader.addBasicPart('6a-1-basic', lambda : testBasic('alphabeta'), 1, description='Tests alphabeta for timeout on smallClassic.')
grader.addHiddenPart('6a-2-hidden', lambda : testHidden('AlphaBetaAgent'), 7, maxSeconds=maxSeconds, description='Tests alphabeta')

# Problem 7: evaluation function

def runq7():
  """
  Runs their expectimax agent a few times and checks for victory!
  """
  random.seed(SEED)
  nGames = 20
  
  agent = submission.choiceAgent()
  print('Running your agent %d times to compute the average score...' % nGames)
  params = '-l smallClassic -p %s -a evalFn=better -q -n %d -c' % (agent, nGames)
  games = pacman.runGames(**pacman.readCommand(params.split(' ')))
  timeouts = [game.agentTimeout for game in games].count(True)
  wins = [game.state.isWin() for game in games].count(True)
  averageScore = sum(game.state.getScore() for game in games) / len(games)
  return timeouts, wins, averageScore

timeouts, wins, averageScore, firstTime = 1024, 0, 0, True

def testq7(thres):
  # We want to use the global values so we only need to compute them once
  global timeouts, wins, averageScore, firstTime

  recordScore = False
  if firstTime:
    firstTime = False
    recordScore = True
    if not grader.fatalError:
      timeouts, wins, averageScore = runq7()

  if timeouts > 0:
    grader.fail('Agent timed out on smallClassic with betterEvaluationFunction. No autograder feedback will be provided.')
  elif wins == 0: 
    grader.fail('Your better evaluation function never won any games.')
  else:
    if averageScore >= thres:
      grader.assignFullCredit()
    if recordScore:
      grader.setSide({'score': averageScore})

grader.addHiddenPart('7a-no-grading', lambda : testq7(0), 0, maxSeconds=300, description='No grading')
grader.addHiddenPart('7a-1-hidden', lambda : testq7(1000), 1, description='Check if score at least 1000 on smallClassic.')
grader.addHiddenPart('7a-2-hidden', lambda : testq7(1200), 2, description='Check if score at least 1200 on smallClassic.')
grader.addHiddenPart('7a-3-hidden', lambda : testq7(1400), 2, description='Check if score at least 1400 on smallClassic.')
grader.addHiddenPart('7a-4-hidden', lambda : testq7(1500), 3, description='Check if score at least 1500 on smallClassic.')

grader.grade()

experimentSoftmaxAlphaSeries(alpha_list=[0.0, 1.0, 10.0], depth=3, nGames=10, layname='smallClassic')
