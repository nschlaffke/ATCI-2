function cnn = cnntrain(cnn, train_x, train_y, opts, val_x, val_y, test_x, test_y)

  assert(isfloat(train_x), 'train_x must be a float');
  assert(nargin == 4 || nargin == 6 || nargin == 8,'number of input arguments must be 4 or 6 or 8')
  opts.validation = 0; if nargin >= 6 && ~isempty(val_x),  opts.validation = 1; end
  opts.test = 0;       if nargin >= 8 && ~isempty(test_x), opts.test = 1;       end

  fhandle = [];
  if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
  end

  m = size(train_x, 3);

  cnn.rL = [];

  disp(['Training CNN for ' num2str(opts.numepochs) ' epochs']);
  for i = 1 : opts.numepochs
    tic;
    kk = randperm(m);
    ini_x = 1;
    end_x = min (m, ini_x + opts.batchsize - 1);
    while ini_x <= m
      batch_x = train_x(:, :, kk(ini_x : end_x), :);
      batch_y = train_y(:,    kk(ini_x : end_x), :);

      cnn = cnnff(cnn, batch_x);
      cnn = cnnbp(cnn, batch_y);
      cnn = cnnapplygrads(cnn, opts);
      if isempty(cnn.rL)
        cnn.rL(1) = cnn.L;
      end
      cnn.rL(end + 1) = 0.99 * cnn.rL(end) + 0.01 * cnn.L;

      ini_x = end_x + 1;
      end_x = min (m, ini_x + opts.batchsize - 1);
    end
    t=toc;

    train_cnntest = cnntest(cnn, train_x, train_y);
    str_loss_train = sprintf('\n  Training set loss = %f',train_cnntest.loss);
    str_acc_train = sprintf('\n  Training set accuracy = %f',1-train_cnntest.e_frac);
    str_loss_val = '';
    str_acc_val = '';
    if opts.validation == 1
      val_cnntest = cnntest(cnn, val_x, val_y);
      str_loss_val = sprintf('  Validation set loss = %f',val_cnntest.loss);
      str_acc_val = sprintf(', Validation set accuracy = %f',1-val_cnntest.e_frac);
    end
    str_loss_test = '';
    str_acc_test = '';
    if opts.test == 1
      test_cnntest = cnntest(cnn, test_x, test_y);
      str_loss_test = sprintf('  Test set loss = %f',test_cnntest.loss);
      str_acc_test = sprintf(', Test set accuracy = %f',1-test_cnntest.e_frac);
    end
    if ishandle(fhandle)
      plot(cnn.rL);
      drawnow;
    end
    disp([' epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds. '...
          str_loss_train str_loss_val str_loss_test str_acc_train str_acc_val str_acc_test]);
  end
    
end
