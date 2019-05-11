
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Training a Deep BackPropagation Network with (optionally) RBM pretraining:
%%%  Every layer is (optionally) pretrained as a Restricted Boltmann Machine
%%%   in a greedy layer-wise unsupervised way with Contrastive Divergence
%%%  Subsequently, the output layer is added and the whole network is trained
%%%   (finetuned) with supervised BackPropagation
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
diary output.txt
diary on
%%% Recursive addpath
ToolboxPath = './ToolboxDeepLearning/';
addpath(genpath(ToolboxPath));

TrainingExamples = {2000, 5000, 10000}; %
EpochsBP = {200}; %
HiddenLayers = {1, 2, 3}; %
HiddenUnits = {50, 200, 500}; %
LearningRateBP = {0.1, 0.01, 0.001}; %
Pretrain = {0, 1}; %
LearningRateRBM = {0.1, 0.01, 0.001}; %
EpochsRBM = {20, 50}; %

% loop without pretraining
for trainingExamples_ = 1:length(TrainingExamples)
    for epochsBP_ = 1:length(EpochsBP)
        for hiddenLayers_ = 1:length(HiddenLayers)
            for hiddenUnits_ = 1:length(HiddenUnits)
                for learningRateBP_ = 1:length(LearningRateBP)
                    
                   % Extract the parameters from the labirynth of MATLAB
                   trainingExamples = TrainingExamples{trainingExamples_}; 
                   epochsBP = EpochsBP{epochsBP_}; 
                   hiddenLayers = HiddenLayers{hiddenLayers_};
                   hiddenUnits = HiddenUnits{hiddenUnits_};
                   learningRateBP = LearningRateBP{learningRateBP_};
                   fprintf('$$$Pretrain: FALSE\ttrainingExamples: %d\tepochsBP: %d\thiddenLayers: %d\thiddenUnits: %d\tlearningRateBP: %.3f\n',... 
                   trainingExamples, epochsBP, hiddenLayers, hiddenUnits, learningRateBP);
                   
                   % Get the data and the network
                   Data = LoadData_MNIST_(trainingExamples);
                   [ReducedData Network] = DeepBPN_RBM_set_parameters_MNIST;
                   
                   % Here we set the parameters
                   Network.PreTrainNetwork_RBM = false;
                   Network.FineTuningBP.NumEpochs = epochsBP;
                   Network.FineTuningBP.LearningRate = learningRateBP;
                   Network.NHiddens = repmat(hiddenUnits, 1, hiddenLayers);
                   ReducedData = trainingExamples;
                   
                   % We train the network
                   DeepBPN_RBM_train_model (Data, ReducedData, Network);
                end
            end
        end
    end 
end
fprintf('PRETRAINGING STARTS\n');
% loop with pretraining
for trainingExamples_ = 1:length(TrainingExamples)
    for epochsBP_ = 1:length(EpochsBP)
        for hiddenLayers_ = 1:length(HiddenLayers)
            for hiddenUnits_ = 1:length(HiddenUnits)
                for learningRateBP_ = 1:length(LearningRateBP)
                    for learningRateRBM_ = 1:length(LearningRateRBM)
                        for epochsRBM_ = 1:length(EpochsRBM)
                    
                           % Extract the parameters from the labirynth of MATLAB
                           trainingExamples = TrainingExamples{trainingExamples_}; 
                           epochsBP = EpochsBP{epochsBP_}; 
                           epochsRBM = EpochsRBM{epochsRBM_};
                           hiddenLayers = HiddenLayers{hiddenLayers_};
                           hiddenUnits = HiddenUnits{hiddenUnits_};
                           learningRateBP = LearningRateBP{learningRateBP_};
                           learningRateRBM = LearningRateRBM{learningRateRBM_}
                           fprintf('$$$Pretrain: TRUE\ttrainingExamples: %d\tepochsBP: %d\thiddenLayers: %d\thiddenUnits: %d\tlearningRateBP: %.3f\n',... 
                           trainingExamples, epochsBP, hiddenLayers, hiddenUnits, learningRateBP);

                           % Get the data and the network
                           Data = LoadData_MNIST_(trainingExamples);
                           [ReducedData Network] = DeepBPN_RBM_set_parameters_MNIST;

                           % Here we set the parameters
                           Network.PreTrainNetwork_RBM = true;
                           Network.PreTrainRBM.NumEpochs = epochsRBM;  
                           Network.PreTrainRBM.LearningRate = learningRateRBM;
                           Network.FineTuningBP.NumEpochs = epochsBP;
                           Network.FineTuningBP.LearningRate = learningRateBP;
                           Network.NHiddens = repmat(hiddenUnits, 1, hiddenLayers);
                           ReducedData = trainingExamples;

                           % We train the network
                           DeepBPN_RBM_train_model (Data, ReducedData, Network);
                        end
                    end
                end
            end
        end
    end 
end

diary off