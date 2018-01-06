# dice-mc
Monte Carlo simulation of dice game

Given a game where you roll a number (d) of dice, reserve any sixes, then re-roll until all dice show sixes: what is the expectation value for the total number of rolls required?

The code computes the answer via the transition matrix Q (where (Q)ij = p(transition from i sixes to j sixes)), then compares the answer with an experimental result from a large number of Monte Carlo trials.

See dice-mc.ipynb
