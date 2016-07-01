# AlphaGoPolicyNet
Implementing the policy network of AlphaGo as the class project of the deep learning class in UCSC. 

AlphaGo Paper: http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html

Convolutional Net to train Go game paper: https://arxiv.org/pdf/1412.3409.pdf

Total Number of Professional Games Downloaded: 85931

Total Number of Training Samples: 17,801,121 (average 207 steps in each game)

####The goal of the project is as following:

At any given time in a game, what's a good next move? I attempt to answer this by looking at moves of professional players. I fed the 17 million professional moves at a known board to a neural network, and try predict what's the best move given a new board. 

Note: The accuracy of a random guess is on average 0.41%. 

####06/01/2016
My first try at the problem: with a fully connected network at 361x100x361 with 100 neurons in the one hidden layer, trained for 60 epoch with 1000 games without parameter tuning, I got a 2.5% accuracy. 


####06/05/2016
with a softmax network at 361x361 with no hidden layer and 1000 games, trained for 60 epoch without parameter tuning, I got a 3.5% validation accuracy. 

####06/06/2016
with network at 361x361x361 with 3000 games, I got a higher 4.0% accuracy

During this whole time the convolutional netork has worse performance than simple vanilla neural network with one hidden layer, possiblity because convolutional net is harder to train.


####07/01/2016
Project finished at June 14th, here are
[The final presentation](https://github.com/MollyZhang/AlphaGoPolicyNet/blob/master/HPC%20and%20Deep%20Learning%20Project%20Presentation.pdf) and 
[The final report](https://github.com/MollyZhang/AlphaGoPolicyNet/blob/master/CMPS218FinalReport%20Molly%20Zhang.pdf).


