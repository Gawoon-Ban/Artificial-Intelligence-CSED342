#!/usr/bin/python

import random
import numpy as np
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: Sentiment Classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE
    feature_vector = {}
    words = x.split() # 공백으로 단어 찢고
    for word in words:
        feature_vector[word] = feature_vector.get(word, 0) + 1 # 각 단어의 횟수 증가시키기
    return feature_vector
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() in 'util' on both trainExamples and
      validationExamples to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE
    def predictor(x):
        score = dotProduct(weights, featureExtractor(x))
        return 1 if score >= 0 else -1
    
    for epoch in range(numEpochs):
        for x,y in trainExamples:
            score = dotProduct(weights, featureExtractor(x))
            if y * score < 1:
                increment(weights, eta * y, featureExtractor(x))
        training_error = evaluatePredictor(trainExamples, predictor)
        validation_error = evaluatePredictor(validationExamples, predictor)
        print("Epoch %d: training error = %f, validation error = %f" % (epoch+1, training_error, validation_error))
    # END_YOUR_CODE

    return weights

############################################################ 
# Problem 4: Multi-layer Perceptron
############################################################ 

class MLPPredictor:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self,input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def forward(self, x):
        """
        Inputs
            x: input feature vector (represented as Phi in Problem description)
        Outputs
            pred: predicted probability (0 to 1)
        """
        # BEGIN_YOUR_ANSWER
        self.x = x # backward에서 사용하기 위해 저장장
        self.z1 = np.dot(x, self.W1) + self.b1 # hidden layer input, x @ W + b 만들기
        self.a1 = self.sigmoid(self.z1) # hidden layer activation, sigmoid 함수 적용
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # output layer input, hidden layer output @ W + b 만들기
        self.a2 = self.sigmoid(self.z2) # output layer activation, sigmoid 함수 적용
        #intermediate activation을 저장해야한다고 했기에 self.z1,z2,a1,a2에 저장
        pred = self.a2
        return pred.flatten() # grader에서 요구하는 형식을 맞추기위해 flatten 사용용
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1)
            target: true label, 0 or 1
        Outputs
            loss: squared loss
        """
        # BEGIN_YOUR_ANSWER 
        return (pred - target) ** 2 # 그냥 squared loss 정의대로 구현
        # END_YOUR_ANSWER 

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        delta2 = 2*(pred - target) * pred * (1 - pred) # output layer delta, sigmoid 함수를 미분한 식을 따라 작성
        delta2 = delta2.reshape(-1, 1) 
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1) # hidden layer delta, sigmoid 함수를 미분한 식을 따라 작성
        dW1 = np.dot(self.x.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER   
        # 걍 값들 구해논거 업데이트 하기
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]
        # END_YOUR_ANSWER  

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 6), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the sqaured loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i:i+1]
                y = Y[i:i+1]
                pred = self.forward(x) # batch size가 1
                loss = self.loss(pred, y)
                gradients = self.backward(pred, y)
                self.update(gradients, learning_rate)
        pred = self.forward(X[-1:])
        answer = self.loss(pred, Y[-1:])
        return answer
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x),4)