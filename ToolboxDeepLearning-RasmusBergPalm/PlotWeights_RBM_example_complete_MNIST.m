
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script must be run AFTER the model 
%%%   DeepBPN_RBM_example_complete_MNIST.m
%%% has been trained
%%% (BPNNet must contain the trained weights)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of pixels of every image
nPixelsX = 28;
nPixelsY = 28;

%%% Plots of the weights of random units of the first layer / Change to see particular weights
nPlots = 24;
nRowsPlot = 4;
nColsPlot = nPlots / nRowsPlot;

% Change to select the set of weights to plot
Weights = DBNNet.rbm{1}.W;  startCol = 1;  %%% bias in a different variable
%Weights = BPNNet.W{1};      startCol = 2;  %%% bias in the first column

nHiddenUnits = size(Weights,1);
rndperm = randperm(nHiddenUnits);
rndidx = rndperm(1:nPlots);

figure; axes('FontSize',16.0); hold on;
colormap('gray');
axis equal;
minImageSC = min(Weights(:));
maxImageSC = max(Weights(:));

for np=1:nPlots
  image = reshape(Weights(rndidx(np),startCol:end),nPixelsX,nPixelsY);
  subplot(nRowsPlot,nColsPlot,np);
  imagesc(image',[minImageSC maxImageSC]);
  title(rndidx(np));
  set(gca,'xtick',[])
  set(gca,'ytick',[])
end;

hold off;
