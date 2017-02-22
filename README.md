# SNLI-Experiments
CS 224N Project, Experiments on the Stanford Natural Language Inference Dataset

TODO:

Infrastructue --
1) Maintain an epoch count to compare with papers. Figure out how to iterate through a split of the data a certain number of times.
2) Correct the accuracy calculation; tensorflow's functions don't work properly across multiple measurements.
3) Give each model configuration a unique id so that the logs are not confused. This should take into account all configuration variables and stay the same across multiple invocations.
4) Figure out how to use Tensorboard. Also mark losses, accuracies, etc., to be collected by the tensorflow summaries.
5) Make padding a model configuration parameter and find an alternative to tf.train.shuffle_batch that allows dynamic padding.
6) Make sure that we can 

Models --
1) The sum of words model from Bowman.
2) Independent rnn encoders on both sentences (fixed and dynamic length)
3) The Rocktaschel paper with attention mechanisms