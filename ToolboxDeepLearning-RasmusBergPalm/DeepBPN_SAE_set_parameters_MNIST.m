
function [ReducedData Network] = DeepBPN_SAE_set_parameters_MNIST;

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
Network.PreTrainNetwork_SAE = true;      %%% true: pre-training with Stacked Denoising AutoEncoders

% Pre-training parameters
Network.PreTrainSAE.IniWRange = 1.0;             %%% Initial weights in Stacked Denoising AutoEncoders pre-training from IniVHRange * Gaussian(0,1) / sqrt(fanin)
Network.PreTrainSAE.NumEpochs = 20;              %%% Number of epochs in Stacked Denoising AutoEncoders pre-training
Network.PreTrainSAE.BatchSize = 100;             %%% Typical values in [10 100]
Network.PreTrainSAE.LearningRate = 0.1;          %%% Learning rate in Stacked Denoising AutoEncoders pre-training
Network.PreTrainSAE.Scaling_learningRate = 1;    %%% Scaling factor for the learning rate (in each epoch) in Stacked Denoising AutoEncoders pre-training
Network.PreTrainSAE.Momentum = 0.8;              %%% Values in [0,1] / Momentum in Stacked Denoising AutoEncoders pre-training
Network.PreTrainSAE.WeightPenaltyL2 = 0;         %%% Weight decay penalty in Stacked Denoising AutoEncoders pre-training
Network.PreTrainSAE.InputZeroFraction = 0.5;     %%% Values in [0,1] / Fraction of inputs set to zero in Stacked *DENOISING* AutoEncoders pre-training
Network.PreTrainSAE.VisualizeWeights = false;    %%% Weight visualization in Stacked Denoising AutoEncoders pre-training

% Fine-tuning parameters
Network.FineTuningBP.IniWRange = 1.0;            %%% Initial NOT PRETRAINED weights in BP fine-tuning from IniWRange * Gaussian(0,1) / sqrt(fanin)
Network.FineTuningBP.NumEpochs = 20;             %%% Number of epochs in BP fine-tuning
Network.FineTuningBP.BatchSize = 100;            %%% Typical values in [10 100]
Network.FineTuningBP.LearningRate = 0.1;         %%% Learning rate in BP fine-tuning
Network.FineTuningBP.Scaling_learningRate = 1;   %%% Scaling factor for the learning rate (in each epoch) in BP fine-tuning
Network.FineTuningBP.Momentum = 0.8;             %%% Values in [0,1] / Momentum in BP fine-tuning
Network.FineTuningBP.WeightPenaltyL2 = 0;        %%% Weight decay penalty in BP fine-tuning
Network.FineTuningBP.DropoutFraction = 0;        %%% Values in [0,1] / Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
Network.FineTuningBP.PlotError = 0;              %%% Plot the error curves (0/1)?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
