
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Training a Deep Convolutional Neural Network with matconvnet
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%%% matconvnet setup
run(fullfile('./matconvnet-1.0-beta20-cpu','matlab','vl_setupnn.m'));
vl_rootnn = fileparts(mfilename('fullpath'));  %%% ERM - current directory

[ReducedData CNNnet CNNopts] = CNNmat_set_parameters_MNIST;

DBimages = LoadData_MNIST(ReducedData);

% ERM - Subsequent adjustments
CNNnet.meta.inputSize = DBimages.meta.inputSize;
CNNnet.meta.classes.name = DBimages.meta.classes;

%%%%%%%%%%%%%%%%%%%%%%%%%%% Train

disp(['Start Training']);
tstart = tic;

[CNNFinalNetwork, InfoStats] = CNNmat_train_model(CNNnet, CNNopts, DBimages);

toc(tstart);
disp(['End Training']);

%%%%%%%%%%%%%%%%%%%%%%%%%%% End train
