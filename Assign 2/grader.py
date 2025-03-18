#!/usr/bin/python3

import graderUtil
import numpy as np
import util
import string
from util import *

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False
############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
    warnings.warn("Must be using Python 3.12 \n")

############################################################
# Problem 1: building intuition
############################################################

grader.add_manual_part('1a', max_points=2, description='simulate SGD')
grader.add_manual_part('1b', max_points=2, description='create small dataset')

############################################################
# Problem 2: predicting movie ratings
############################################################

grader.add_manual_part('2a', max_points=2, description='loss')
grader.add_manual_part('2b', max_points=3, description='gradient')
grader.add_manual_part('2c', max_points=3, description='smallest magnitude')

############################################################
# Problem 3: sentiment classification
############################################################

### 3a

# Basic sanity check for feature extraction
def test3a0():
    ans = {"a":2, "b":1}
    grader.require_is_equal(ans, submission.extractWordFeatures("a b a"))
grader.add_basic_part('3a-0-basic', test3a0, max_seconds=1, description="basic test")

def test3a1():
    random.seed(42)
    sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
    submission_ans = submission.extractWordFeatures(sentence)
    if solution_exist:
        solution_ans = solution.extractWordFeatures(sentence)
        grader.require_is_equal(solution_ans, submission_ans)
grader.add_hidden_part('3a-1-hidden', test3a1, max_seconds=1, description="test multiple instances of the same word in a sentence")

### 3b

def test3b0():
    trainExamples = [("hello world", 1), ("goodnight moon", -1)]
    testExamples = [("hello", 1), ("moon", -1)]
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numEpochs=20, eta=0.01)
    grader.require_is_greater_than(0, weights["hello"])
    grader.require_is_less_than(0, weights["moon"])
grader.add_basic_part('3b-0-basic', test3b0, max_seconds=3, description="basic sanity check for learning correct weights on two training and testing examples each")

def test3b1():
    trainExamples = [("hi bye", 1), ("hi hi", -1)]
    testExamples = [("hi", -1), ("bye", 1)]
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numEpochs=20, eta=0.01)
    grader.require_is_less_than(0, weights["hi"])
    grader.require_is_greater_than(0, weights["bye"])
grader.add_basic_part('3b-1-basic', test3b1, max_seconds=2, description="test correct overriding of positive weight due to one negative instance with repeated words")

def test3b2():
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
    grader.require_is_less_than(0.04, trainError)
    grader.require_is_less_than(0.30, validationError)
grader.add_basic_part('3b-2-basic', test3b2, max_points=2, max_seconds=16, description="test classifier on real polarity dev dataset")




############################################################
# Problem 4: MLP
############################################################

def test4a0():
    random.seed(SEED)
    np.random.seed(SEED)
    X, _ = generateExample()
    mlp = submission.MLPPredictor(6,3,1)
    answer = np.array([0.32259345,0.28290188,0.31240478,0.57001625])
    grader.require_is_equal(answer, mlp.forward(X), 0.0001)
grader.add_basic_part('4a-0-basic', test4a0, max_seconds=1, description="test forward function on Problem2 dataset")

def test4a1():
    def get_gen():
        for _ in range(10):
            random.seed(SEED)
            np.random.seed(SEED)
            mlp = submission.MLPPredictor(2,5,1)
            test_data = np.random.rand(100, 2)
            pred = mlp.forward(test_data)
            if solution_exist:
                mlp2 = solution.MLPPredictor(2,5,1)
                answer = mlp2.forward(test_data)
                yield grader.require_is_equal(answer,pred, 0.0001)
            else:
                yield True
    all(get_gen())
grader.add_hidden_part('4a-1-hidden', test4a1, max_seconds=1, description="test forward function on random data")


### 4b (loss and backward functions)
##### 4b-0 (loss function)
def test4b0():
    X, Y = generateExample()
    random.seed(SEED)
    np.random.seed(SEED)
    mlp = submission.MLPPredictor(6,3,1)
    pred = mlp.forward(X)
    loss = mlp.loss(pred, Y)
    answer = np.array([0.10406653, 0.08003348, 0.47278718, 0.18488602])
    grader.require_is_equal(answer, loss, 0.0001)
grader.add_basic_part('4b-0-basic', test4b0, max_seconds=1, description="test loss function on Problem1 dataset")

##### 4b-1 (backward function)
def test4b1():
    X, Y = generateExample()
    random.seed(SEED)
    np.random.seed(SEED)
    mlp = submission.MLPPredictor(6,3,1)
    pred = mlp.forward(X)
    gradients = mlp.backward(pred, Y)
    answer = {
        "W1": np.array([[ 0.03747391,  0.0175744 , -0.02007225],
                       [ 0.01080103,  0.0739285 , -0.0736167 ],
                       [-0.01032213, -0.03676712,  0.04172597],
                       [ 0.0267785 ,  0.09257349, -0.09599611],
                       [-0.01597748, -0.01864499,  0.02237942],
                       [ 0.04779604,  0.05434152, -0.06179822]]),
        "b1": np.array([[ 0.04827494,  0.0915029 , -0.09368895]]),
        "W2": np.array([[-0.14173446],
                       [-0.05920717],
                       [-0.17061063]]),
        "b2": np.array([[-0.25040362]])
    }
    grader.require_is_equal(answer, gradients, 0.0001)
grader.add_basic_part('4b-1-basic', test4b1, max_seconds=2, description="test backward function on Problem1 dataset")

### 4b-2 (hidden)
def test4b2():
    def get_gen():
        for _ in range(10):
            random.seed(SEED)
            np.random.seed(SEED)
            mlp = submission.MLPPredictor(2,5,1)
            test_X = np.random.rand(100, 2)
            test_Y = np.random.randint(0, 2, 100)
            pred = mlp.forward(test_X)
            gradients = mlp.backward(pred, test_Y)
            if solution_exist:
                mlp_answer = solution.MLPPredictor(2,5,1)
                pred_answer = mlp_answer.forward(test_X)
                gradients_answer = mlp_answer.backward(pred_answer, test_Y)
                yield grader.require_is_equal(gradients, gradients_answer, 0.0001)
            else:
                yield True
    all(get_gen())
grader.add_hidden_part('4b-2-hidden', test4b2, max_seconds=1, description="test backward function on random data")



### 4c  (update and train functions)
def test4c0():
    X, Y = generateExample()
    random.seed(SEED)
    np.random.seed(SEED)
    mlp = submission.MLPPredictor(6,3,1)
    grader.require_is_equal(0.001603244388550324, mlp.train(X, Y, epochs=1000, learning_rate=0.1), 0.0001)
grader.add_basic_part('4c-0-basic', test4c0, max_seconds=10, description="test train function on Problem1 dataset")

def test4c1():
    def get_gen():
        for _ in range(10):
            random.seed(SEED)
            np.random.seed(SEED)
            mlp = submission.MLPPredictor(2,5,1)
            test_X = np.random.rand(100, 2)
            test_Y = np.random.randint(0, 2, 100)
            loss = mlp.train(test_X, test_Y, epochs=10, learning_rate=0.1)
            if solution_exist:
                mlp_answer = solution.MLPPredictor(2,5,1)
                loss_answer = mlp_answer.train(test_X, test_Y, epochs=10, learning_rate=0.1)
                yield grader.require_is_equal(loss, loss_answer, 0.0001)
            else:
                yield True
    all(get_gen())
grader.add_hidden_part('4c-1-hidden', test4c1, max_seconds=10, description="test train function on random data")

grader.grade()
