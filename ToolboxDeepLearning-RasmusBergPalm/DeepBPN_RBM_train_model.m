
function [DBNNet BPNNet] = DeepBPN_RBM_train_model (Data, ReducedData, Network)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DO NOT MODIFY FROM HERE !!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Start Training']);

NHiddens = Network.NHiddens;

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

%%% Needed in all cases
BPNNopts.iniWRange = Network.FineTuningBP.IniWRange;               %%% PREVIOUS TO dbnunfoldtonn/nnsetup !!!

%%% Pre-training
if Network.PreTrainNetwork_RBM
  DBNNet.sizes = NHiddens;
  DBNNopts.alpha = Network.PreTrainRBM.LearningRate;               %%% PREVIOUS TO dbnsetup !!!
  DBNNopts.momentum = Network.PreTrainRBM.Momentum;                %%% PREVIOUS TO dbnsetup !!!
  DBNNopts.weightPenaltyL2 = Network.PreTrainRBM.WeightPenaltyL2;  %%% PREVIOUS TO dbnsetup !!!
  DBNNopts.iniVHRange = Network.PreTrainRBM.IniVHRange;            %%% PREVIOUS TO dbnsetup !!!
  DBNNopts.inputDataType = Network.PreTrainRBM.InputDataType;      %%% PREVIOUS TO dbnsetup !!!
  DBNNet = dbnsetup(DBNNet, Data.train_x, DBNNopts);
  DBNNopts.numepochs = Network.PreTrainRBM.NumEpochs;
  DBNNopts.batchsize = Network.PreTrainRBM.BatchSize;
  DBNNet = dbntrain(DBNNet, Data.train_x, DBNNopts);
  if Network.PreTrainRBM.VisualizeWeights
    figure; visualize(DBNNet.rbm{1}.W');
  end;
  % Unfold DBNNet to BPNNet
  BPNNet = dbnunfoldtonn(DBNNet, NOutputs, BPNNopts);
else
  BPNNet = nnsetup([NInputs NHiddens NOutputs], BPNNopts);
end;

%%% Fine-tuning
BPNNet.activation_function = 'sigm';
BPNNet.output = 'softmax';
BPNNet.learningRate = Network.FineTuningBP.LearningRate;
BPNNet.scaling_learningRate = Network.FineTuningBP.Scaling_learningRate;
BPNNet.momentum = Network.FineTuningBP.Momentum;
BPNNet.weightPenaltyL2 = Network.FineTuningBP.WeightPenaltyL2;
BPNNet.dropoutFraction = Network.FineTuningBP.DropoutFraction;
BPNNet.nonSparsityPenalty  = 0;      %%%  Fixed non sparsity penalty (sparsity not used)
BPNNet.sparsityTarget = 0.05;        %%%  Fixed sparsity target
BPNNet.inputZeroMaskedFraction = 0;  %%%  Only used for Denoising AutoEncoders (percentage of masked 0s)
BPNNopts.numepochs = Network.FineTuningBP.NumEpochs;
BPNNopts.batchsize = Network.FineTuningBP.BatchSize;
BPNNopts.plot = Network.FineTuningBP.PlotError;
BPNNet = nntrain(BPNNet, Data.train_x, Data.train_y, BPNNopts, Data.val_x, Data.val_y, Data.test_x, Data.test_y);

toc(tstart);

disp(['End Training']);

%%%%%%%%%%%%%%%%%%%%%%%%%%% End train
