function nn  = nntrain(nn, train_x, train_y, opts, val_x, val_y, test_x, test_y)
%NNTRAIN trains a neural net
% nn = nntrain(nn, x, y, opts, ...) trains the neural network nn with
% input x and output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b)

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6 || nargin == 8,'number of input arguments must be 4 or 6 or 8')

loss.train.e = []; loss.train.e_frac = [];
loss.val.e   = []; loss.val.e_frac   = [];
loss.test.e  = []; loss.test.e_frac  = [];
opts.validation = 0; if nargin >= 6 && ~isempty(val_x),  opts.validation = 1; end
opts.test = 0;       if nargin >= 8 && ~isempty(test_x), opts.test = 1;       end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

numepochs = opts.numepochs;
batchsize = opts.batchsize;

disp(['Training NN ' ' (' num2str(nn.size) ') with BackPropagation for ' num2str(numepochs) ' epochs (batchsize: ' num2str(batchsize) ')']);
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    ini_x = 1;
    end_x = min (m, ini_x + batchsize - 1);
    while ini_x <= m
        batch_x = train_x(kk(ini_x : end_x), :);
        batch_y = train_y(kk(ini_x : end_x), :);
        real_batchsize = size(batch_x,1);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        ini_x = end_x + 1;
        end_x = min (m, ini_x + batchsize - 1);
    end
    
    t = toc;

    lossdata = nneval(nn, train_x, train_y);
    loss.train.e(end+1) = lossdata.e;
    str_perf = sprintf('\n  Full-batch training loss = %f', loss.train.e(end));
    str_acc = '';
    if isfield(lossdata,'e_frac') && ~isempty(lossdata.e_frac)
      loss.train.e_frac(end+1) = lossdata.e_frac;
      str_acc = sprintf('\n  Training set accuracy = %f', 1-loss.train.e_frac(end));
    end
    %
    if opts.validation == 1
      lossdata = nneval(nn, val_x, val_y);
      loss.val.e(end+1) = lossdata.e;
      str_perf = sprintf('%s, validation loss = %f', str_perf, loss.val.e(end));
      if isfield(lossdata,'e_frac') && ~isempty(lossdata.e_frac)
        loss.val.e_frac(end+1) = lossdata.e_frac;
        str_acc = sprintf('%s, Validation set accuracy = %f', str_acc, 1-loss.val.e_frac(end));
      end
    end
    %
    if opts.test == 1
      lossdata = nneval(nn, test_x, test_y);
      loss.test.e(end+1) = lossdata.e;
      str_perf = sprintf('%s, test loss = %f', str_perf, loss.test.e(end));
      if isfield(lossdata,'e_frac') && ~isempty(lossdata.e_frac)
        loss.test.e_frac(end+1) = lossdata.e_frac;
        str_acc = sprintf('%s, Test set accuracy = %f', str_acc, 1-loss.test.e_frac(end));
      end
    end
    %
    disp([' epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds. '...
          str_perf str_acc]);
    if isinf(loss.train.e(end)) || isnan(loss.train.e(end))
      disp(' *** nntrain cancelled: Invalid weights ***'); return;
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end

