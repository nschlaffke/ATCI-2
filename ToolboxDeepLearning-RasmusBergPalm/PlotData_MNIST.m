
clear all;

Data = LoadData_MNIST;

% Number of pixels of every image
nPixelsX = 28;
nPixelsY = 28;

%%% Random plots of train_x / Change to see particular images
nPlots = 24;
nRowsPlot = 4;
nColsPlot = nPlots / nRowsPlot;

rndperm = randperm(size(Data.train_x,1));
rndidx = rndperm(1:nPlots);

figure; axes('FontSize',16.0); hold on;
colormap('gray');
axis equal;

for np=1:nPlots
  image = reshape(Data.train_x(rndidx(np),:),nPixelsX,nPixelsY);
  label = find(Data.train_y(rndidx(np),:)==1)-1;
  subplot(nRowsPlot,nColsPlot,np);
  imagesc(image',[min(image(:)) max(image(:))]);
  %imagesc(image',[0 1]);
  title(label);
  set(gca,'xtick',[])
  set(gca,'ytick',[])
end;

hold off;

