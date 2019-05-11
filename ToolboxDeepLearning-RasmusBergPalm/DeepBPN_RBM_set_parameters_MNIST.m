
function [ReducedData Network] = DeepBPN_RBM_set_parameters_MNIST;

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
ReducedData = 10000;                     %%% If ReducedData > 0, it uses 'ReducedData' traininig input vectors for training

% Architecture
Network.NHiddens = [200 200 200];        %%% [100] / [500 500] / etc (sigmoid hidden units / softmax output units)

% Pre-training y/n
Network.PreTrainNetwork_RBM = true;      %%% true: pre-training with Stacked RBMs

% Pre-training parameters
Network.PreTrainRBM.InputDataType = 'binary';    %%% Input type: binary/gaussian
Network.PreTrainRBM.IniVHRange = 1.0;            %%% Initial weights in Stacked RBMs pre-training from IniVHRange * Gaussian(0,1) / sqrt(fanin)
Network.PreTrainRBM.NumEpochs = 20;              %%% Number of epochs in Stacked RBMs pre-training
Network.PreTrainRBM.BatchSize = 100;             %%% Typical values in [10 100]
Network.PreTrainRBM.LearningRate = 0.1;          %%% Learning rate in Stacked RBMs pre-training
Network.PreTrainRBM.Momentum = 0.8;              %%% Momentum in Stacked RBMs pre-training
Network.PreTrainRBM.WeightPenaltyL2 = 0.0002;    %%% Weight decay penalty in RBMs pre-training
Network.PreTrainRBM.VisualizeWeights = false;    %%% Weight visualization in RBMs pre-training

% Fine-tuning parameters
Network.FineTuningBP.IniWRange = 1.0;            %%% Initial NOT PRETRAINED weights in BP fine-tuning from IniWRange * Gaussian(0,1) / sqrt(fanin)
Network.FineTuningBP.NumEpochs = 20;             %%% Number of epochs in BP fine-tuning
Network.FineTuningBP.BatchSize = 100;            %%% Typical values in [10 100]
Network.FineTuningBP.LearningRate = 0.1;         %%% Learning rate in BP fine-tuning
Network.FineTuningBP.Scaling_learningRate = 1;   %%% Scaling factor for the learning rate (in each epoch) in BP fine-tuning
Network.FineTuningBP.Momentum = 0.8;             %%% Values in [0,1] / Momentum in BP fine-tuning
Network.FineTuningBP.WeightPenaltyL2 = 0;        %%% Weight decay penalty in BP fine-tuning
Network.FineTuningBP.DropoutFraction = 0;        %%% Values in [0,1] / Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
Network.FineTuningBP.PlotError = 0  ;              %%% Plot the error curves (0/1)?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
