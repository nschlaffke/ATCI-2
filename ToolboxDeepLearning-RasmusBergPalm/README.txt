
If you want to change the data set, edit LoadData_MNIST.m

If you want to visualize the MNIST inputs, run PlotData_MNIST.m

For training a BackPropagation network for classification
(with sigmoid activation hidden units) optionally pretrained with
Restricted Boltzmann Machines:
  - Edit DeepBPN_RBM_set_parameters_Dataset.m
  - Run DeepBPN_RBM_example_complete_Dataset.m

For training a BackPropagation network for classification
(with sigmoid activation hidden units) optionally pretrained with
Denoising AutoEncoders:
  - Edit DeepBPN_SAE_set_parameters_Dataset.m
  - Run DeepBPN_SAE_example_complete_Dataset.m

For training a standard BackPropagation network for classification
(with sigmoid/tanh_opt/reclinear activation hidden units):
  - Edit DeepBPN_MLP_set_parameters_Dataset.m
  - Run DeepBPN_MLP_example_complete_Dataset.m

If you want to visualize the weights after training the model in
DeepBPN_RBM_example_complete_MNIST.m, run
PlotWeights_RBM_example_complete_MNIST.m
