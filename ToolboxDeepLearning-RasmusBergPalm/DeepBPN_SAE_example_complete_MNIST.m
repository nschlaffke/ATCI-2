
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Train a Deep BackPropagation Network with Denoising AutoEncoder pretraining:
%%%  Every layer is (optionally) pretrained as a Denoising AutoEncoder
%%%   in a greedy layer-wise unsupervised way with BackPropagation
%%%  Subsequently, the output layer is added and the whole network is trained
%%%   (finetuned) with supervised BackPropagation
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%%% Recursive addpath
ToolboxPath = './ToolboxDeepLearning/';
addpath(genpath(ToolboxPath));

Data = LoadData_MNIST;

[ReducedData Network] = DeepBPN_SAE_set_parameters_MNIST;

[SAENet BPNNet] = DeepBPN_SAE_train_model (Data, ReducedData, Network);
