function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    if strcmp(opts.inputDataType,'binary')
      assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    end
    m = size(x, 1);

    for i = 1 : opts.numepochs
        tic;
        kk = randperm(m);
        err = 0;
        ini_x = 1;
        end_x = min (m, ini_x + opts.batchsize - 1);
        while ini_x <= m
            batch = x(kk(ini_x : end_x), :);
            real_batchsize = size(batch,1);
            
            v1 = batch;
            h1 = sigmrnd(repmat(rbm.c', real_batchsize, 1) + v1 * rbm.W');
            if     strcmp(opts.inputDataType,'binary')
              %%% (ERM important change) v2 = sigmrnd(repmat(rbm.b', real_batchsize, 1) + h1 * rbm.W);
              v2 = sigm(repmat(rbm.b', real_batchsize, 1) + h1 * rbm.W);
            elseif strcmp(opts.inputDataType,'gaussian')
              v2 = repmat(rbm.b', real_batchsize, 1) + h1 * rbm.W;
            else error('rbmtrain: inputDataType not implemented');
            end;
            h2 = sigm(repmat(rbm.c', real_batchsize, 1) + v2 * rbm.W');

            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * ( (c1 - c2) - opts.weightPenaltyL2 * rbm.vW ) / real_batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / real_batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / real_batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v1 - v2) .^ 2));

            ini_x = end_x + 1;
            end_x = min (m, ini_x + opts.batchsize - 1);
        end
        t = toc;
        disp([' epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds.'...
              ' Average reconstruction error is: ' num2str(err / m)]);
        if isinf(err) || isnan(err)
           disp(' *** rbmtrain cancelled: Invalid weights ***'); return;
        end
    end
end
