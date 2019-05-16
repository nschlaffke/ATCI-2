
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Training a Deep Multilayer Perceptron with BackPropagation
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%%% Recursive addpath
ToolboxPath = './ToolboxDeepLearning/';
addpath(genpath(ToolboxPath));

Data = LoadData_MNIST;

TrainingExamples = {2000, 5000, 10000};
LearningRates = {0.1, 0.01, 0.001};
Dropouts = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
HiddenLayers = {1, 2, 3};
HiddenUnits = {100, 200, 500};

diary log.txt;

for te=1:length(TrainingExamples)
    for lr=1:length(LearningRates)
        for d=1:length(Dropouts)
            for hl=1:length(HiddenLayers)
                for hu=1:length(HiddenUnits)
                    training_examples = cell2mat(TrainingExamples(te));
                    learning_rate = cell2mat(LearningRates(lr));
                    dropout = cell2mat(Dropouts(d));
                    hidden_layers = cell2mat(HiddenLayers(hl));
                    hidden_units = cell2mat(HiddenUnits(hu));
                    fprintf('Params %d %4.3f %4.2f %d %d\n',training_examples,learning_rate,dropout,hidden_layers,hidden_units);
                    
                    [ReducedData Network] = MyDeepBPN_MLP_set_parameters_MNIST(training_examples,learning_rate,dropout,hidden_layers,hidden_units);
                    BPNNet = MyDeepBPN_MLP_train_model (Data, ReducedData, Network);
                    fprintf('-----\n');
                end
            end
        end
    end
end

diary off

% [ReducedData Network] = MyDeepBPN_MLP_set_parameters_MNIST;

% BPNNet = MyDeepBPN_MLP_train_model (Data, ReducedData, Network);
