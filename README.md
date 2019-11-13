The value of print(score) is [0.038257713587692754, 0.9915]

1.	Convolution is convolving the entire input image with a filter and get a feature as an output.
2.	Kernel is the matrix that will extract the particular type of feature from the input.
3.	No of epochs is the number of times our model has been trained on the dataset. 
4.	1x1 convolutions are used to decrease the output channels.
5.	3x3 convolution is best as the parameters required in it are substantially low as compared to other higher values of convolution. Two since it is odd, it helps maintaining the symmetry on all the sides. Also every 3x3 can be a 2x2 if that is what is required in the model, he backprop will make sure of it.
6.	Output of a kernel gives specific features and since there are different kernels in each layer the collective output is called a Feature Map.
7.	Receptive field of any layer is the part of the input image it is able to see and thus receptive field of the last layer must be at least of the size of the object we are looking at.
