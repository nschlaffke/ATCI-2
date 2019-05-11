
function imdb = LoadData_MNIST (ReducedData);

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

%train_x = train_x/255;
%val_x = val_x/255;
%test_x = test_x/255;

%%% Reduced Data
if ReducedData > 0
  r = randperm(size(train_x,1));
  train_x = train_x(r(1:ReducedData),:);
  train_y = train_y(r(1:ReducedData),:);
end;

%%% Auxiliar variables
NInputs = size(train_x,2);
NOutputs = size(train_y,2);
NExamples = size(train_x,1);

%%% Show information
disp(['Number of training examples: ' num2str(NExamples)]);
disp(['Number of inputs: ' num2str(NInputs) '  Number of outputs: ' num2str(NOutputs)]);

% --- Specific adjustments for matconvnet

x1=train_x;
x1=permute(reshape(x1,size(x1,1),28,28),[2 3 1]);

x2=test_x;
x2=permute(reshape(x2,size(x2,1),28,28),[2 3 1]);

y1=code1ofCtolab1dim(train_y)';

y2=code1ofCtolab1dim(test_y)';

% If set == {1/2/3}: {train/validation/set}
set = [ones(1,numel(y1)) 2*ones(1,numel(y2))];  %%% ERM - In this case we use the test data as validation

dim1 = size(x1,1);
dim2 = size(x1,2);
dim3 = 1;            %%% ERM - Number of input channels (change if needed)

data = single(reshape(cat(3, x1, x2),dim1,dim2,dim3,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

%mean(mean(mean(data)))

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2);
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);
%%% ERM - Added for clarity
imdb.meta.inputSize = [dim1 dim2 dim3];

return;

% --------------------------------------------------------------------
function lab1dim = code1ofCtolab1dim(y);
% --------------------------------------------------------------------

[N NOut] = size(y);

MaxY = max(y,[],2);
lab1dim = zeros(N,1);
for i=1:N
  for j=1:NOut
    if y(i,j) == MaxY(i)
      jmax = j;
    end;
  end;
  lab1dim(i) = jmax; %%% Labels are 1,2,3...
end;

return;

