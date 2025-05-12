#!/usr/bin/env python3
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""

import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.add_hidden_part = grader.add_basic_part
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

############################################################

def get_csp_result(csp, BacktrackingSearch=None, **kargs):
    if BacktrackingSearch is None:
        BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                              submission.BacktrackingSearch)
    solver = BacktrackingSearch()
    solver.solve(csp, **kargs)
    return (solver.optimalWeight,
            solver.numOptimalAssignments,
            solver.numOperations)

############################################################
# Problem 2a: N-Queens

def test2a_1():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.require_is_equal(1.0, nQueensSolver.optimalWeight)
    grader.require_is_equal(92, nQueensSolver.numOptimalAssignments)
    grader.require_is_equal(2057, nQueensSolver.numOperations)

grader.add_basic_part('2a-1-basic', test2a_1, 1, max_seconds=1,
        description="Basic test for create_nqueens_csp for n=8")

def test2a_2():
    pred = get_csp_result(submission.create_nqueens_csp(3))
    if solution_exist:
        grader.require_is_equal(get_csp_result(solution.create_nqueens_csp(3)), pred)

grader.add_hidden_part('2a-2-hidden', test2a_2, 2, max_seconds=1,
        description="Test create_nqueens_csp with n=3")

def test2a_3():
    pred1 = get_csp_result(submission.create_nqueens_csp(4))
    if solution_exist:
        grader.require_is_equal(get_csp_result(solution.create_nqueens_csp(4)), pred1)
    
    pred2 = get_csp_result(submission.create_nqueens_csp(7))
    if solution_exist:
        grader.require_is_equal(get_csp_result(solution.create_nqueens_csp(7)), pred2)

grader.add_hidden_part('2a-3-hidden', test2a_3, 2, max_seconds=1,
        description="Test create_nqueens_csp with different n")

############################################################
# Problem 2b: Most constrained variable


def test2b_1():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.require_is_equal(1.0, mcvSolver.optimalWeight)
    grader.require_is_equal(92, mcvSolver.numOptimalAssignments)
    grader.require_is_equal(1361, mcvSolver.numOperations)

grader.add_basic_part('2b-1-basic', test2b_1, 1, max_seconds=1,
        description="Basic test for MCV with n-queens CSP")

def test2b_2():
    # We will use our implementation of n-queens csp
    # mcvSolver.solve(our_nqueens_csp(8), mcv = True)
    create_nqueens_csp = (solution.create_nqueens_csp if solution_exist else
                          submission.create_nqueens_csp)
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(create_nqueens_csp(8), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.require_is_equal(answer, pred)

grader.add_hidden_part('2b-2-hidden', test2b_2, 2, max_seconds=1,
        description="Test for MCV with n-queens CSP")

def test2b_3():
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(util.create_map_coloring_csp(), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.require_is_equal(answer, pred)

grader.add_hidden_part('2b-3-hidden', test2b_3, 2, max_seconds=1,
        description="Test MCV with different CSPs")

############################################################
# Problem 3b: Sum factor

def test3b_1():
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    csp.add_unary_factor(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.require_is_equal(4, sumSolver.numOptimalAssignments)

    csp.add_unary_factor(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.require_is_equal(2, sumSolver.numOptimalAssignments)

grader.add_basic_part('3b-1-basic', test3b_1, 1, max_seconds=1, description="Basic test for get_sum_variable")

def test3b_2():
    BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                          submission.BacktrackingSearch)

    def get_result(get_sum_variable):
        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        csp.add_unary_factor(sumVar, lambda n: n > 0)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.require_is_equal(get_result(solution.get_sum_variable), pred)

grader.add_hidden_part('3b-2-hidden', test3b_2, 2, max_seconds=1, description="Test get_sum_variable with empty list of variables")

def test3b_3():
    def get_result(get_sum_variable):
        csp = util.CSP()
        csp.add_variable('A', [0, 1, 2])
        csp.add_variable('B', [0, 1, 2])
        csp.add_variable('C', [0, 1, 2])

        sumVar = get_sum_variable(csp, 'sum-up-to-7', ['A', 'B', 'C'], 7)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp.add_unary_factor(sumVar, lambda n: n == 6)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.require_is_equal(get_result(solution.get_sum_variable), pred)

grader.add_hidden_part('3b-3-hidden', test3b_3, 2, max_seconds=1, description="Test get_sum_variable with different variables")

grader.grade()
