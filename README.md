# nlp-a2-N-grams
1) The data directory contains 2 csvs, one is the unfiltered dataset given originally, and one after all the pre-processing.
2) pre-processing.py contains the code for pre-processing the comments in a parallelized fashion.
3) Utils.py contains some utility functions that are used in the N-gram language model.
4) ngrams.py is the file that contains the implementation of the N-gram language model class and its methods.
5) models.py is the main experiment file where we instantiate the model for different n values and calculate the perplexity and log(perplexity).
6) plotting.py is used to plot the perplexity values for inference and analysis.
7) Smoothing_Comparison.txt stores the result of models.py which is a comparison between perplexities of different smoothing techniques on n-gram models.
8) The repo also contains the final documentation of the assignment in the pdf format.
