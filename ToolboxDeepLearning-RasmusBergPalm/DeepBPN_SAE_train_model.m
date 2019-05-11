
function [SAENet BPNNet] = DeepBPN_SAE_train_model (Data, ReducedData, Network)

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

%%% Pre-training
if Network.PreTrainNetwork_SAE
  SAENopts.iniWRange = Network.PreTrainSAE.IniWRange;    %%% PREVIOUS TO saesetup !!!
  SAENet = saesetup([NInputs NHiddens],SAENopts);
  for i = 1 : numel(NHiddens)
    SAENet.ae{i}.activation_function = 'sigm';
    SAENet.ae{i}.output = 'sigm';
    SAENet.ae{i}.learningRate = Network.PreTrainSAE.LearningRate;
    SAENet.ae{i}.scaling_learningRate = Network.PreTrainSAE.Scaling_learningRate;
    SAENet.ae{i}.momentum = Network.PreTrainSAE.Momentum;
    SAENet.ae{i}.weightPenaltyL2 = Network.PreTrainSAE.WeightPenaltyL2;
    SAENet.ae{i}.dropoutFraction = 0;
    SAENet.ae{i}.nonSparsityPenalty  = 0;      %%%  Fixed non sparsity penalty (sparsity not used)
    SAENet.ae{i}.sparsityTarget = 0.05;        %%%  Fixed sparsity target
    SAENet.ae{i}.inputZeroMaskedFraction = Network.PreTrainSAE.InputZeroFraction;
  end;
  SAENopts.numepochs = Network.PreTrainSAE.NumEpochs;
  SAENopts.batchsize = Network.PreTrainSAE.BatchSize;
  SAENet = saetrain(SAENet, Data.train_x, SAENopts);
  if Network.PreTrainSAE.VisualizeWeights
    visualize(SAENet.ae{1}.W{1}(:,2:end)')
  end;
  % Use the SDAE to initialize a FFNN
  BPNNopts.iniWRange = Network.FineTuningBP.IniWRange;              %%% PREVIOUS TO nnsetup !!!
  BPNNet = nnsetup([NInputs NHiddens NOutputs], BPNNopts);
  for i = 1 : numel(NHiddens)
    BPNNet.W{i} = SAENet.ae{i}.W{1};
  end;
else
  BPNNopts.iniWRange = Network.FineTuningBP.IniWRange;              %%% PREVIOUS TO nnsetup !!!
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
