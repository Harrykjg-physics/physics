# landscape-aggregation


Simplified python model based on 'A Landscape Theory of Aggregation' (Robert Axelrod and D. Scott Bennett, BJPS 1993)

PLEASE NOTE This is not an attempt to replicate the paper!

Paper abstract: "Aggregation means the organization of elements of a system into patterns that tend to put highly compatible elements together and less compatible elements apart. Landscape theory predicts how aggregation will lead to alignments among actors (such as nations), whose leaders are myopic in their assessments and incremental in their actions. The predicted configurations are based upon the attempts of actors to minimize their frustration based upon their pairwise propensities to align with some actors and oppose others. These attempts lead to a local minimum in the energy landscape of the entire system. The theory is supported by the results of two cases: the alignment of seventeen European nations in the Second World War..."

The following python implementation makes a number of simplifications but remains true to the general outline of the model as presented in the paper, seeking to calculate the most stable aggregation in 1938. The focus is on the method more than the data - for example, I have so far not been able to include ethnicity in a meaningful way - treating it in the same way as religion will not work (each country has a different ethnic majority so all propensities would be +1, with no information content).

Possible aggregations (distinct ways of splitting the 17 countries into exactly two sides) are represented by two vectors representing the respective sides (A and B). Pairwise propensities to cooperation or conflict are represented as a matrix, with data from The Penguin Atlas of World History Vol II and The Correlates of War Project*

For each possible aggregation, a total energy is calculated as the sum of the total (within group) frustration on each of the two opposing sides - the most stable / lowest energy aggregation is recorded. For example, it the propensity matrix is labelled P, and the two sides for a particular aggregation are represented by vectors a and b (where T denotes the transpose), the total 'energy' or degree of frustration is calculated as Etot = (aT)Pa +(bT)Pb

The implementation is inspired by the mothod of calculating expectation values in QM in that the frustration or energy for each of the two sides is calculated as <state_vector|Matrix_operator|state_vector>.

At present the result is a predicted stable aggregation with 5/17 countries placed in the wrong alliance - as opposed to just one or two (depending on the year) in the original paper. With 12/17 countries correctly placed, I put the probability of getting this close by chance at 9%. Not there yet.

Possibilities for future improvements:

- Better data, including ethnic conflict - saved in a data file to avoid hard coding
- More sophisticated handling of religious conflict, as in the original paper - looking at the respective proportions of all major religious groups in both countries for each dyad
- Calculation of local minima rather than one absolute minimum

*As in the original paper - supplemented with data from Wikipedia if necessary (national religion data) - since only the dominant religion is recorded, and most countries were close to religious monocultures at that time, the information on Wikipedia is sufficient for the simplified implementation used in my code.
