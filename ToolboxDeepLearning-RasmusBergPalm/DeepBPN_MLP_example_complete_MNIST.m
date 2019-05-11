
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Training a Deep Multilayer Perceptron with BackPropagation
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%%% Recursive addpath
ToolboxPath = './ToolboxDeepLearning/';
addpath(genpath(ToolboxPath));

Data = LoadData_MNIST;

[ReducedData Network] = DeepBPN_MLP_set_parameters_MNIST;

BPNNet = DeepBPN_MLP_train_model (Data, ReducedData, Network);
