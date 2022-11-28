%% free exploration analysis main script

addpath('\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\free exploration data')
addpath('\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\my_functions')

PlotIT = 0;

% load data
load('ndata_s01_stroke.mat')

% create figure
if PlotIt == 1
figure
hold on

%plot 1D position
plot((1:length(ndata))*.005, ndata(:,1),'b') % plot x-position
xlabel('Time (seconds)')
ylabel('robot position')
% plot((1:length(ndata))*.005, ndata(:,2),'r') % plot x-position
end

%filter position data - 5th order butterworth
sHz = 200;
nth = 5;
cHz = 12; 
[ndata_smoothed(:,1)] = butterx(ndata(:,1), sHz, nth, cHz);
[ndata_smoothed(:,2)] = butterx(ndata(:,2), sHz, nth, cHz);

% plot 
p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,1),'r'); % plot x-position

s = surf(zzz.X,zzz.Y,(zzz.Z/sum(zzz.Z(:))));
set(s,'LineStyle','none')
view(2)
hold on

% differentiate
[ndata_smoothed(:,3), ndata_smoothed(:,5)] = dbl_diff(ndata_smoothed(:,1),sHz); % differentiate x
[ndata_smoothed(:,4), ndata_smoothed(:,6)] = dbl_diff(ndata_smoothed(:,2),sHz); % differentiate y


figure
plot(ndata_smoothed(:,1),ndata_smoothed(:,2),'.')


% build 2D histogram
nbins = 40;
zzz = histcount2(ndata_smoothed(:,1:2), nbins, [-.3 -.25], [.3 .1]);

% create heatmap
figure
% s = surf(zzz.X,zzz.Y,(zzz.Z/sum(zzz.Z(:))));
s = surf(zzz.X,zzz.Y,zzz.Z);

set(s,'LineStyle','none')
set(gca,'clim',[0 2000])

view(2)
hold on
