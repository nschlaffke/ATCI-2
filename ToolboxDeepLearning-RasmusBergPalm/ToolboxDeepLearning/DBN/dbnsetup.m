function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha           = opts.alpha;
        dbn.rbm{u}.momentum        = opts.momentum;
        dbn.rbm{u}.weightPenaltyL2 = opts.weightPenaltyL2;
	%%% First layer has inputType as the data; the rest are binary
	if u == 1
	  dbn.rbm{u}.inputType = opts.inputDataType;
	else
	  dbn.rbm{u}.inputType = 'binary';
	end

        % (ERM change) dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.W  = opts.iniVHRange * randn(dbn.sizes(u + 1), dbn.sizes(u)) / sqrt(dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
