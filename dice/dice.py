# I play a lot of poker dice and wanted to know the expectation value for how
# many rolls it would take to get all aces (sixes) from the five dice, if at
# each round aces are reserved (i.e. without replacement).

# I wrote the MC code before knowing how to compute the answer analytically,
# but I've now added that method as a check/comparison.

import numpy
import scipy.special


def prob(d, psuccess, a, b):
    # transition probabilities from a sixes to b sixes given d dice
    p = psuccess
    q = 1.0 - p

    r = d - a 			# number of dice left to roll
    s = b - a			# number of successes required
    f = d - b			# number of failures required

    C = scipy.special.comb(r, s)    # number of ways to get s sixes from r dice

    if (b < a):
        pi = 0
    elif (b == a):
        pi = (q**f)
    elif (b > a):
        pi = C * (p**s) * (q**f)

    return pi       			# transition probability from a to b sixes


def cp(d, psuccess, a, b):
    # cumulative probability from current number of sixes (a) to b sixes
    cp = 0
    for i in range(a, b+1):
        cp += prob(d, psuccess, a, i)
    return cp


def trial(d, psuccess):
    sixes = 0 			        # number of sixes
    rollcount = 0 			    # number of rolls

    while (sixes < d):
        # run trial until all dice show sixes
        rollcount += 1
        rand = numpy.random.random()
        for i in range(sixes, d+1):
            if rand >= cp(d, psuccess, sixes, i-1):
                temp = i
        sixes = temp

    return rollcount

print

def simulation(trials, d, psuccess):
    results = []

    for i in range(trials):
        rollcount = trial(d, psuccess)
        results.append(rollcount)
        if i % (trials/10) == 0:
            print 'loading...', 100 * i / trials, '%'

    resultsarray = numpy.array(results)

    mean = numpy.mean(resultsarray)
    sd = numpy.std(resultsarray)
    stderror = sd / (trials**0.5)
    return mean, stderror


def transitionmatrix(d, psuccess):
    Q = numpy.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Q[i, j] = prob(d, psuccess, i, j)
    return Q


def ex_analyticsolution(d, psuccess):
    Q = transitionmatrix(d, psuccess)
    Id = numpy.identity(d)
    N = numpy.linalg.inv(Id - Q)
    onesV = numpy.ones(d)
    EXV = numpy.dot(N, onesV)
    EX = EXV[0]
    return EX


# Parameters
d = 5                   # number of dice
psuccess = (1.0/6.0)    # probability of rolling a six for a single die
trials = 50000          # number of trials

# Main Program
EXA = ex_analyticsolution(d, psuccess)
print '\nI have', d, 'dice. I roll them all, reserving any sixes. How many rolls to get all sixes?'
print '\nAnalytic solution (transition matrix):\nExpectation value for number of rolls required =', EXA, '\n'

print 'Monte Carlo simulation using', trials, 'trials for determining expectation value'

mean, stderror = simulation(trials, d, psuccess)
print 'Simulation complete\n'
print 'Experimental results after', trials, 'trials:'
print 'Expectation value for number of rolls required =', mean, '+/-', stderror, '\n'

if abs(mean - EXA) < (2*stderror):
    print 'This is consistent with the analytic solution.\n'

