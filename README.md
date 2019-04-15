# NLU

This is a gender analysis of the emails in the Enron dataset using Python and NLTK.

The research employs a fastText classification to investigate whether a model trained to identify gender in workplace emails generalizes to identify gender in employee performance reviews. Several studies have looked at the problem of gender classification in workplace communications by utilizing the publicly available email dataset that was released when the Enron Corporation went bankrupt, and have achieved accuracies as high as 95% with classifiers trained on variations of a stylometric feature space. We extend this line of research to evaluate (i) how the fastText classifier performs on these tasks, and (ii) whether the learning of the fastText classifier transfers to gender prediction in other occupational datasets, i.e. when performance is tested on a dataset of performance reviews. We conducted six ex- periments, using word-based, sentence-based, and document-based features. 

The dataset can be found here: https://www.kaggle.com/wcukierski/enron-email-dataset/data
