#!/usr/bin/env python3

import random
import collections
import graderUtil

grader = graderUtil.Grader()

# load the student's solution module
submission = grader.load('submission')

try:
    import solution
    SEED = solution.SEED
    solution_exists = True
except ModuleNotFoundError:
    SEED = 42
    solution_exists = False

grader.useSolution = solution_exists

############################################################
# Problem 5a: find_alphabetically_first_word

grader.add_basic_part(
    '5a-0-basic', 
    lambda: grader.require_is_equal(
        'alphabetically', 
        submission.find_alphabetically_first_word('which is the first word alphabetically')
    ),
    max_points=2,
    description='5a simple test case'
)

grader.add_basic_part(
    '5a-1-basic',
    lambda: grader.require_is_equal(
        'cat', 
        submission.find_alphabetically_first_word('cat sun dog')
    ),
    description='5a simple test case'
)

grader.add_basic_part(
    '5a-2-basic', 
    lambda: grader.require_is_equal(
        '0', 
        submission.find_alphabetically_first_word(' '.join(str(x) for x in range(100000)))
    ), 
    description='5a big test case'
)

############################################################
# Problem 5b: euclidean_distance

grader.add_basic_part(
    '5b-0-basic', 
    lambda: grader.require_is_equal(5, submission.euclidean_distance((1, 5), (4, 1))),
    description='5b simple test case'
)

def test5b1():
    random.seed(SEED)
    for _ in range(100):
        x1 = random.randint(0, 10)
        y1 = random.randint(0, 10)
        x2 = random.randint(0, 10)
        y2 = random.randint(0, 10)
        ans2 = submission.euclidean_distance((x1, y1), (x2, y2))
        if solution_exists:
            grader.require_is_equal(ans2, solution.euclidean_distance((x1, y1), (x2, y2)))
        
grader.add_hidden_part('5b-1-hidden', test5b1, max_points=2, description='5b 100 random trials')

############################################################
# Problem 5c: sparse_vector_dot_product

def test5c0():
    grader.require_is_equal(
        15, 
        submission.sparse_vector_dot_product(
            collections.defaultdict(float, {'a': 5}),
            collections.defaultdict(float, {'b': 2, 'a': 3})
        )
    )

grader.add_basic_part('5c-0-basic', test5c0, max_points=2, description='5c simple test')

def randvec_sparse():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) + 5
    return v

def test5c1():
    random.seed(SEED)
    for _ in range(10):
        v1 = randvec_sparse()
        v2 = randvec_sparse()
        ans2 = submission.sparse_vector_dot_product(v1, v2)
        if solution_exists:
            grader.require_is_equal(ans2, solution.sparse_vector_dot_product(v1, v2))

grader.add_hidden_part('5c-1-hidden', test5c1, max_points=2, description='5c random trials')

############################################################
# Problem 5d: increment_sparse_vector

def test5d0():
    v = collections.defaultdict(float, {'a': 5})
    submission.increment_sparse_vector(v, 2, collections.defaultdict(float, {'b': 2, 'a': 3}))
    grader.require_is_equal(
        collections.defaultdict(float, {'a': 11, 'b': 4}),
        v
    )

grader.add_basic_part('5d-0-basic', test5d0, max_points=2, description='5d simple test')

def test5d1():
    random.seed(SEED)
    for _ in range(10):
        v1a = randvec_sparse()
        v1b = v1a.copy()
        v2 = randvec_sparse()
        submission.increment_sparse_vector(v1b, 4, v2)
        # remove keys with zero value for comparison
        for key in list(v1b):
            if v1b[key] == 0:
                del v1b[key]
        if solution_exists:
            v1c = v1a.copy()
            submission.increment_sparse_vector(v1c, 4, v2)
            for key in list(v1c):
                if v1c[key] == 0:
                    del v1c[key]
            grader.require_is_equal(v1b, v1c)

grader.add_hidden_part('5d-1-hidden', test5d1, max_points=2, description='5d random trials')

############################################################
# Problem 5e: find_nonsingleton_words

def test5e0():
    grader.require_is_equal(
        {'the', 'fox'},
        submission.find_nonsingleton_words('the quick brown fox jumps over the lazy fox')
    )

grader.add_basic_part('5e-0-basic', test5e0, description='5e simple test')

def test5e1(num_tokens, num_types):
    random.seed(SEED)
    text = ' '.join(str(random.randint(0, num_types)) for _ in range(num_tokens))
    ans2 = submission.find_nonsingleton_words(text)
    if solution_exists:
        grader.require_is_equal(ans2, solution.find_nonsingleton_words(text))

grader.add_hidden_part('5e-1-hidden', lambda: test5e1(1000, 10), max_points=1, description='5e random trials')
grader.add_hidden_part('5e-2-hidden', lambda: test5e1(10000, 100), max_points=1, description='5e random trials (bigger)')

############################################################
# Problem 5f: mutate_sentences

def test5f0():
    grader.require_is_equal(
        sorted(['a a a a a']),
        sorted(submission.mutate_sentences('a a a a a'))
    )
    grader.require_is_equal(
        sorted(['the cat']),
        sorted(submission.mutate_sentences('the cat'))
    )
    grader.require_is_equal(
        sorted(['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']),
        sorted(submission.mutate_sentences('the cat and the mouse'))
    )

grader.add_basic_part('5f-0-basic', test5f0, max_points=2, description='5f simple test')

def gen_sentence(alphabet_size, length):
    return ' '.join(str(random.randint(0, alphabet_size)) for _ in range(length))

def test5f1():
    random.seed(SEED)
    for _ in range(10):
        sentence = gen_sentence(3, 5)
        ans2 = submission.mutate_sentences(sentence)
        if solution_exists:
            grader.require_is_equal(ans2, solution.mutate_sentences(sentence))

grader.add_hidden_part('5f-1-hidden', test5f1, max_points=2, description='5f random trials')

def test5f2():
    random.seed(SEED)
    for _ in range(10):
        sentence = gen_sentence(25, 10)
        ans2 = submission.mutate_sentences(sentence)
        if solution_exists:
            grader.require_is_equal(ans2, solution.mutate_sentences(sentence))

grader.add_hidden_part('5f-2-hidden', test5f2, max_points=2, description='5f random trials (bigger)')

############################################################
grader.grade()
