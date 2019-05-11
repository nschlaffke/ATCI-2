
function Data = LoadData_MNIST;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load MNIST data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DataPath = '../Data/';
Filename = 'mnist_uint8.mat';
DataSet = fullfile(DataPath,Filename);
disp(['Loading data set: ' DataSet]);
load(DataSet)

train_x = double(train_x);  train_y = double(train_y);
val_x   = [];               val_y   = [];
test_x  = double(test_x);   test_y  = double(test_y);

train_x = train_x/255;
val_x = val_x/255;
test_x = test_x/255;

Data.train_x = train_x;
Data.train_y = train_y;
Data.val_x = val_x;
Data.val_y = val_y;
Data.test_x = test_x;
Data.test_y = test_y;

return;
