This repo contain class to perform pairwise comparisons of objects and
transforation this annotation to ansolute value rank for each object

Updates performed according to http://personal.psu.edu/drh20/papers/bt.pdf

There are two preposed ways of sampling pairs: 

* "Dense" Where number of pairs = number of combinations of objects
* "Sparse" Where connectivity graph is checked each comparison step. Sampling perfrom in stochastic manner, considering the object "popularity" in sampling probability.
