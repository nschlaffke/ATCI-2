function nn = nnsetup(architecture, opts)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    existOpts = false;
    if exist('opts','var'), existOpts = true; end;

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers:
                                                        %   'sigm' (sigmoid), 'tanh_opt' (optimal tanh) or 'reclinear' (ReLU)
    nn.learningRate                     = 0.1;          %  learning rate
    nn.momentum                         = 0.8;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

    for i = 2 : nn.n   
        % weights initialization
        if existOpts && isfield(opts,'iniWRange')
           %%% (ERM change)
           nn.W{i - 1} = opts.iniWRange * randn(nn.size(i), nn.size(i - 1)+1) / sqrt(nn.size(i - 1));
        else
           %%% Original version
           nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        end;
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end