
function BPNNet = DeepBPN_MLP_train_model (Data, ReducedData, Network)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DO NOT MODIFY FROM HERE !!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Start Training']);

NHiddens = Network.NHiddens;
ActFunction = Network.ActFunction;

%%% Reduced Data
if ReducedData > 0
  r = randperm(size(Data.train_x,1));
  Data.train_x = Data.train_x(r(1:ReducedData),:);
  Data.train_y = Data.train_y(r(1:ReducedData),:);
end;

%%% Auxiliar variables
NInputs = size(Data.train_x,2);
NOutputs = size(Data.train_y,2);
NExamples = size(Data.train_x,1);

%%% Show information
disp(['Number of training examples: ' num2str(NExamples)]);
disp(['Number of inputs: ' num2str(NInputs) '  Number of outputs: ' num2str(NOutputs)]);
NHiddenLayers = length(NHiddens);
disp(['Number of hidden layers: ' num2str(NHiddenLayers) ' (' num2str(NHiddens) ')']);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Train

tstart = tic;

BPNNopts.iniWRange = Network.TrainBP.IniWRange;    %%% PREVIOUS TO nnsetup !!!
BPNNet = nnsetup([NInputs NHiddens NOutputs], BPNNopts);

%%% Rest of parameters
BPNNet.activation_function = ActFunction;
BPNNet.output = 'softmax';
BPNNet.learningRate = Network.TrainBP.LearningRate;
BPNNet.scaling_learningRate = Network.TrainBP.Scaling_learningRate;
BPNNet.momentum = Network.TrainBP.Momentum;
BPNNet.weightPenaltyL2 = Network.TrainBP.WeightPenaltyL2;
BPNNet.dropoutFraction = Network.TrainBP.DropoutFraction;
BPNNet.nonSparsityPenalty  = 0;      %%%  Fixed non sparsity penalty (sparsity not used)
BPNNet.sparsityTarget = 0.05;        %%%  Fixed sparsity target
BPNNet.inputZeroMaskedFraction = 0;  %%%  Only used for Denoising AutoEncoders (percentage of masked 0s)
BPNNopts.numepochs = Network.TrainBP.NumEpochs;
BPNNopts.batchsize = Network.TrainBP.BatchSize;
BPNNopts.plot = Network.TrainBP.PlotError;
BPNNet = nntrain(BPNNet, Data.train_x, Data.train_y, BPNNopts, Data.val_x, Data.val_y, Data.test_x, Data.test_y);

toc(tstart);

disp(['End Training']);

%%%%%%%%%%%%%%%%%%%%%%%%%%% End train
