
function [ReducedData CNNnet CNNopts] = CNNmat_set_parameters_MNIST;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Global parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Setting parameters']);

% Random seed
RandomSeed = sum(100*clock);             %%% sum(100*clock) / A fixed value allows to replicate exactly the same results
rng('default');
rng(RandomSeed);

% Data
ReducedData = 10000;                     %%% If ReducedData > 0, it uses 'ReducedData' traininig input vectors for training

% Architecture
IniWRange = 1/100;                       %%% Initial weights range from IniWRange * Gaussian(0,1)
CNNnet.layers = {};                      %%% See http://www.vlfeat.org/matconvnet/mfiles/simplenn/vl_simplenn/
                                         %%%  for an explanation of the parameters

CNNnet.layers{end+1} = struct('type', 'conv', ...       %%% Convolution with  20 filters of (Width x Height x Depth) 5x5x1 with stride 1 and no padding
                              'weights', {{IniWRange*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                              'stride', 1, 'pad', 0);   %%% Input (Width x Height x Depth): 28x28x1 / Output: 24x24x20

CNNnet.layers{end+1} = struct('type', 'pool', ...       %%% Max-pooling 2x2 with stride 2 and no padding
                              'method', 'max', ...      %%% Input (Width x Height x Depth): 24x24x20 / Output: 12x12x20
                              'pool', [2 2], 'stride', 2, 'pad', 0);

CNNnet.layers{end+1} = struct('type', 'conv', ...       %%% Convolution with  50 filters of (Width x Height x Depth) 5x5x20 with stride 1 and no padding
                              'weights', {{IniWRange*randn(5,5,20,50, 'single'), zeros(1,50,'single')}}, ...
                              'stride', 1, 'pad', 0);   %%% Input (Width x Height x Depth): 12x12x20 / Output: 8x8x50

CNNnet.layers{end+1} = struct('type', 'pool', ...       %%% Max-pooling 2x2 with stride 2 and no padding
                              'method', 'max', ...      %%% Input (Width x Height x Depth): 8x8x50 / Output: 4x4x50
                              'pool', [2 2], 'stride', 2, 'pad', 0);

CNNnet.layers{end+1} = struct('type', 'conv', ...       %%% Convolution with 500 filters of (Width x Height x Depth) 4x4x50 with stride 1 and no padding
                              'weights', {{IniWRange*randn(4,4,50,500, 'single'), zeros(1,500,'single')}}, ...
                              'stride', 1, 'pad', 0);   %%% Input (Width x Height x Depth): 4x4x50 / Output: 1x1x500  *** FULLY CONNECTED LAYER ***

CNNnet.layers{end+1} = struct('type', 'relu');          %%% Rectified linear non-linearity Input/Output: 1x1x500

CNNnet.layers{end+1} = struct('type', 'conv', ...       %%% Convolution with 500 filters of (Width x Height x Depth) 1x1x500 with stride 1 and no padding
                              'weights', {{IniWRange*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
                              'stride', 1, 'pad', 0);   %%% Input (Width x Height x Depth): 1x1x500 / Output: 1x1x500  *** FULLY CONNECTED LAYER ***

CNNnet.layers{end+1} = struct('type', 'softmaxloss');   %%% Softmax (output) layer

% Training parameters
CNNopts.networkType = 'simplenn';            %%% Do not modify! ('dagnn' would look for 'cnn_train_dag', but it is not adjusted)
CNNopts.errorFunction = 'multiclass';        %%% 'binary'/'multiclass'
CNNopts.batchNormalization = true;           %%% 'true' for inserting batch normalization layers
CNNopts.batchNormalizationLayers = [1,4,7];  %%% Indexes where insert batch normalization layers
CNNopts.batchSize = 100;                     %%% Typical values in [10 100]
CNNopts.gpus = [];                           %%% Ids of the GPUs (if any)
CNNopts.numEpochs = 10;                      %%% Number of epochs of the training
CNNopts.learningRate = 0.0001;               %%% Learning rate of the training
CNNopts.weightDecay = 0.0005;                %%% Weight decay penalty of the training
CNNopts.momentum = 0.9;                      %%% Values in [0,1] / Momentum of the training
CNNopts.randomSeed = RandomSeed;             %%% Do not modify!
CNNopts.cudnn = true;                        %%% 'true' for using the cudnn libraries
CNNopts.verboseLevel = 0;                    %%% More verbose for larger values
CNNopts.plotStatistics = true;               %%% 'true' for plotting the evolution of the training
