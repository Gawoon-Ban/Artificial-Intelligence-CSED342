import copy
from util import CSP, get_or_variable
from typing import Dict, List


############################################################
# Problem 2

def create_nqueens_csp(n: int = 8) -> CSP:
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_factor().
    """
    csp = CSP()
    # Add n variables, each representing a queen in a row with domain as column positions
    for i in range(n):
        csp.add_variable(i, list(range(n)))
    
    # Add binary factors to ensure no two queens threaten each other
    for i in range(n):
        for j in range(i + 1, n):
            csp.add_binary_factor(i, j, lambda x, y: (
                x != y and  # Not in same column
                abs(x - y) != abs(i - j)  # Not on same diagonal
            ))
    return csp

# A backtracking algorithm that solves weighted CSP.
class BacktrackingSearch:
    def reset_results(self) -> None:
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0
        # Keep track of the number of optimal assignments and assignments.
        self.numOptimalAssignments = 0
        self.numAssignments = 0
        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0
        # Keep track of the number of operations to get to the very first successful assignment.
        self.firstAssignmentNumOperations = 0
        # List of all solutions found.
        self.allAssignments = []
        self.allOptimalAssignments = []

    def print_stats(self) -> None:
        if self.optimalAssignment:
            print(f'Found {self.numOptimalAssignments} optimal assignments \
                    with weight {self.optimalWeight} in {self.numOperations} operations')
            print(f'First assignment took {self.firstAssignmentNumOperations} operations')
        else:
            print("No consistent assignment to the CSP was found. The CSP is not solvable.")

    def get_delta_weight(self, assignment: Dict, var, val) -> float:
        assert var not in assignment
        w = 1.0
        if self.csp.unaryFactors[var]:
            w *= self.csp.unaryFactors[var][val]
            if w == 0:
                return w
        for var2, factor in list(self.csp.binaryFactors[var].items()):
            if var2 not in assignment:
                continue
            w *= factor[val][assignment[var2]]
            if w == 0:
                return w
        return w

    def satisfies_constraints(self, assignment: Dict, var, val) -> bool:
        return self.get_delta_weight(assignment, var, val) != 0

    def solve(self, csp: CSP, mcv: bool = False, ac3: bool = False) -> None:
        self.csp = csp
        self.mcv = mcv
        self.ac3 = ac3
        self.reset_results()
        self.domains = {var: list(self.csp.values[var]) for var in self.csp.variables}
        self.backtrack({}, 0, 1)
        self.print_stats()

    def backtrack(self, assignment: Dict, numAssigned: int, weight: float) -> None:
        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            self.numAssignments += 1
            newAssignment = {var: assignment[var] for var in self.csp.variables}
            self.allAssignments.append(newAssignment)
            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                    self.allOptimalAssignments.append(newAssignment)
                else:
                    self.numOptimalAssignments = 1
                    self.allOptimalAssignments = [newAssignment]
                self.optimalWeight = weight
                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        var = self.get_unassigned_variable(assignment)
        ordered_values = self.domains[var]
        if not self.ac3:
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    del assignment[var]
        else:
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    localCopy = copy.deepcopy(self.domains)
                    self.domains[var] = [val]
                    self.apply_arc_consistency(var)
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    self.domains = localCopy
                    del assignment[var]

    def get_unassigned_variable(self, assignment: Dict):
        if not self.mcv:
            for var in self.csp.variables:
                if var not in assignment:
                    return var
        else:
            min_count = float('inf')
            selected_var = None
            for var in self.csp.variables:
                if var not in assignment:
                    valid_count = 0
                    for val in self.domains[var]:
                        if self.satisfies_constraints(assignment, var, val):
                            valid_count += 1
                    if valid_count < min_count or (valid_count == min_count and 
                        self.csp.variables.index(var) < self.csp.variables.index(selected_var 
                            if selected_var is not None else self.csp.variables[0])):
                        min_count = valid_count
                        selected_var = var
            return selected_var

    def apply_arc_consistency(self, var) -> None:
        def remove_inconsistent_values(var1, var2):
            removed = False
            factor = self.csp.binaryFactors[var1][var2]
            for val1 in list(self.domains[var1]):
                if (self.csp.unaryFactors[var1] and self.csp.unaryFactors[var1][val1] == 0) or \
                        all(factor[val1][val2] == 0 for val2 in self.domains[var2]):
                    self.domains[var1].remove(val1)
                    removed = True
            return removed

        queue = [var]
        while len(queue) > 0:
            curr = queue.pop(0)
            for neighbor in self.csp.get_neighbor_vars(curr):
                if remove_inconsistent_values(neighbor, curr):
                    queue.append(neighbor)


def create_sum_variable(csp: CSP, name: str, variables: List, maxSum: int) -> tuple:
    result = ('sum', name, 'aggregated')
    csp.add_variable(result, list(range(maxSum + 1)))
    if len(variables) == 0:
        csp.add_unary_factor(result, lambda x: x == 0)
        return result
    domain = []
    for i in range(maxSum + 1):
        for j in range(i, maxSum + 1):
            domain.append((i, j))
    for i in range(len(variables)):
        csp.add_variable(('sum', name, str(i)), domain)
    csp.add_unary_factor(('sum', name, '0'), lambda x: x[0] == 0)
    for i in range(len(variables)):
        f = ('sum', name, str(i))
        csp.add_binary_factor(f, variables[i], lambda x, y: x[1] == x[0] + y)
    for i in range(len(variables) - 1):
        f0 = ('sum', name, str(i))
        f1 = ('sum', name, str(i + 1))
        csp.add_binary_factor(f0, f1, lambda x, y: x[1] == y[0])
    csp.add_binary_factor(
        ('sum', name, str(len(variables) - 1)), result, lambda x, y: x[1] == y)
    return result

############################################################
# Problem 3b

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain range(0, maxSum+1), such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.
    """
    # BEGIN_YOUR_CODE
    result = ('sum', name, 'aggregated')
    csp.add_variable(result, list(range(maxSum + 1)))

    if len(variables) == 0:
        csp.add_unary_factor(result, lambda x: x == 0)
        return result

    if len(variables) == 1:
        csp.add_binary_factor(variables[0], result, lambda x, y: x == y)
        return result

    # Create auxiliary variables with tuple domains to track partial sums
    domain = [(i, j) for i in range(maxSum + 1) for j in range(i, maxSum + 1)]
    for i in range(len(variables)):
        csp.add_variable(('sum', name, str(i)), domain)

    # Constraint for the first auxiliary variable
    csp.add_unary_factor(('sum', name, '0'), lambda x: x[0] == 0)

    # Binary factors to propagate sums
    for i in range(len(variables)):
        f = ('sum', name, str(i))
        csp.add_binary_factor(f, variables[i], lambda x, y: x[1] == x[0] + y)

    # Binary factors to link auxiliary variables
    for i in range(len(variables) - 1):
        f0 = ('sum', name, str(i))
        f1 = ('sum', name, str(i + 1))
        csp.add_binary_factor(f0, f1, lambda x, y: x[1] == y[0])

    # Link last auxiliary variable to result
    csp.add_binary_factor(('sum', name, str(len(variables) - 1)), result, lambda x, y: x[1] == y)

    return result
    # END_YOUR_CODE