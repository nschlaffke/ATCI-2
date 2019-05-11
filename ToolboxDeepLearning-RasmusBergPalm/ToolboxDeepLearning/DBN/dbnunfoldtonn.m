function nn = dbnunfoldtonn(dbn, outputsize, opts)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added.
    if exist('outputsize','var')
        size = [dbn.sizes outputsize];
    else
        size = [dbn.sizes];
    end
    if exist('opts','var')
        nn = nnsetup(size, opts);
    else
        nn = nnsetup(size);
    end;
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W];
    end
end

