# MNIST Digit Classifier

This handwritten digit classifier was built with Tensorflow, Numpy, and Matplotlib to implement a Convolutional Neural Network (CNN).

It has 3 main files: 
1. The file `model.py` contains the implementation of the tensorflow functions to process the MNIST image dataset and train the neural network.
2. The file `model_on_gui.py` allows a user to run it and try it out on a GUI. You just draw the digit, press 'classify' to see the result, and press 'clear' to draw another digit. It will refit the image, run the model that was trained from `model.py` and then output the result.
3. The file `plot_results.py` allows a user to plot the resulting outputs of the neural network on the test image set. 