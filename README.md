# AlphaGoValueNet
Implementing the value network of Alpha Go as the class project of the deep learning class in UCSC. 

AlphaGo Paper: http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html

Total Number of Professional Games Used: 85931

Total Number of Training Samples: 17,801,121 (average 207 steps in each game)

The goal of my project is as following:

At any given time in a game, what's a good next move? I attempt to answer this by looking at moves of professional players. I fed the 17 million professional moves at a known board to a neural network, and try predict what's the best move given a new board. 

The accuracy of a random guess is somewhere between 0.28% to 1%, averaging 0.41%. 

My first try at the problem: with a fully connected network at 361x100x361 with 100 neurons in the one hidden layer, trained for 60 epoch without parameter tuning, I got a 2.5% accuracy. 

