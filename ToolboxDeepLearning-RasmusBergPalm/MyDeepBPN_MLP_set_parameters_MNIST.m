
function [ReducedData Network] = MyDeepBPN_MLP_set_parameters_MNIST(training_examples,learning_rate,dropout,hidden_layers,hidden_units);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Global parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Setting parameters']);

% Random seed and Random initialization
RandomSeed = sum(100*clock);             %%% sum(100*clock) / A fixed value allows to replicate exactly the same results
% 
rand('state',RandomSeed);
randn('state',RandomSeed);

% Data
ReducedData = training_examples;                     %%% If ReducedData > 0, it uses 'ReducedData' traininig input vectors for training

% Architecture
if hidden_layers == [1]
    Network.NHiddens = [hidden_units];
elseif hidden_layers == [2]
    Network.NHiddens = [hidden_units hidden_units];
elseif hidden_layers == [3]
    Network.NHiddens = [hidden_units hidden_units hidden_units];
end

% Network.NHiddens = [200 200 200];        %%% [100] / [500 500] / etc (softmax output units)
Network.ActFunction = 'reclinear';        %%% 'sigm' (sigmoid), 'tanh_opt' (optimal tanh) or 'reclinear' (ReLU)

% Training parameters
Network.TrainBP.IniWRange = 1.0;            %%% Initial weights in BP training from IniWRange * Gaussian(0,1) / sqrt(fanin)
Network.TrainBP.NumEpochs = 30;             %%% Number of epochs in BP training
Network.TrainBP.BatchSize = 100;            %%% Typical values in [10 100]
Network.TrainBP.LearningRate = learning_rate;         %%% Learning rate in BP training
Network.TrainBP.Scaling_learningRate = 1;   %%% Scaling factor for the learning rate (in each epoch) in BP training
Network.TrainBP.Momentum = 0.8;             %%% Values in [0,1] / Momentum in BP training
Network.TrainBP.WeightPenaltyL2 = 0;        %%% Weight decay penalty in BP training
Network.TrainBP.DropoutFraction = dropout;        %%% Values in [0,1] / Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
Network.TrainBP.PlotError = 0;              %%% Plot the error curves (0/1)?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%