# AlphaGoValueNet
Implementing the value network of Alpha Go as the class project of the deep learning class in UCSC. 

AlphaGo Paper: http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html

Total Number of Professional Games Used: 85931

Total Number of Training Samples: 17,801,121 (average 207 steps in each game)


The gist of my project is as following:

At any given time in a game, what's a good next move? This question is answered by looking at historical professional games. I fed the 17 million steps that professional players took in the 85 thousand games, and train a neural network to predict what's the best move to make in a new board not in training example. 

The accruacy of the predictions are calculated by test examples where I do know what move professional players made. On average, the accuracy of a random guess is somewhere between 1/361 to 1/50, depending on how many availabe moves there are on board. And what's the accuracy of the machine learning prediction? Well this project is set up to find that out as the goal. 

