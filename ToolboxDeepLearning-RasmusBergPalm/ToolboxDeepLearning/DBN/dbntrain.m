function dbn = dbntrain(dbn, x, opts)
    numepochs = opts.numepochs;
    batchsize = opts.batchsize;

    n = numel(dbn.rbm);

    i = 1;
    if     strcmp(opts.inputDataType,'binary')
      disp(['Training binary-binary RBM in layer ' num2str(i) ' (' num2str(size(dbn.rbm{i}.W')) ') with CD1 for ' num2str(numepochs) ' epochs (batchsize: ' num2str(batchsize) ')']);
    elseif strcmp(opts.inputDataType,'gaussian')
      disp(['Training gaussian-binary RBM in layer ' num2str(i) ' (' num2str(size(dbn.rbm{i}.W')) ') with CD1 for ' num2str(numepochs) ' epochs (batchsize: ' num2str(batchsize) ')']);
    else error('dbntrain: inputDataType not implemented');
    end
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        disp(['Training binary-binary RBM in layer ' num2str(i) ' (' num2str(size(dbn.rbm{i}.W')) ') with CD1 for ' num2str(numepochs) ' epochs (batchsize: ' num2str(batchsize) ')']);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end

end
