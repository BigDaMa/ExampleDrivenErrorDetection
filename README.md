# Example-Driven Error Detection

Traditional error detection approaches require user-defined parameters and rules. 
Thus, the user has to know both, the error detection system and the data. 
However, we can also formulate error detection as a semi-supervised classification problem that only requires domain expertise. 
The challenges for such an approach are twofold: to represent the data in a way that enables a classification algorithm to identify various kinds of data errors, and to pick the most promising data values for validation.
In this paper, we address these challenges with our new example-driven error detection method (ED2). 
First, we discuss and identify the appropriate features to locate different kinds of data errors across different data types. 
Second, we present a new two-dimensional multi-classifier sampling strategy for active learning. 
The combined application of these techniques enables the convergence of the classification task with high detection accuracy. 
On datasets with 3% to 35% error rates, ED2 outperforms existing error detection approaches with, on average, only 1% labels per dataset.

# Using ED2
To run the experiments, first, you need to set the paths in a configuration file with the name of your machine. Examples can be found here: ~/model/ml/configuration/resources/

Then, you can adapt the file ~/model/ml/experiments/features_experiment_multi.py to run the experiments that you are interested in.
