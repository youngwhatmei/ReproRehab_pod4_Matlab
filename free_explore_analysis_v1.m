%% free_explore_analysis.m
% exercise on filtering, differentiating, plotting time-series data, creating 2D heatmaps
% ndata includes planar (x,y) robot end point motion data recorded from stroke and healthy subjects
% performing exploratory movement


%%

%Add paths to free exploration data and useful functions
addpath('\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\free exploration data')
addpath('\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\my_functions')


PlotIT = 0; % if PlotIt == 1, plot

% load data
load('ndata_s01_stroke.mat')


% create figure
if PlotIt == 1
figure
hold on
end


%plot 1D x, y position time-series
if PlotIt == 1
plot((1:length(ndata))*.005, ndata(:,1),'b') % plot x-position
xlabel('Time (seconds)')
ylabel('robot position')
% plot((1:length(ndata))*.005, ndata(:,2),'r') % plot y-position
end

% filter x, y position data using 5th order butterworth filter
sHz = 200; % sampling rate
nth = 5; % filter order
cHz = 12; % cutoff frequency
[ndata_smoothed(:,1)] = butterx(ndata(:,1), sHz, nth, cHz);
[ndata_smoothed(:,2)] = butterx(ndata(:,2), sHz, nth, cHz);

% plot filtered x, y position data 
if PlotIt == 1
p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,1),'r'); % plot x-position
% p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,2),'r'); % plot y-position
end


% differentiate filtered x, y position data for x,y velocity and acceleration
[ndata_smoothed(:,3), ndata_smoothed(:,5)] = dbl_diff(ndata_smoothed(:,1),sHz); % differentiate x
[ndata_smoothed(:,4), ndata_smoothed(:,6)] = dbl_diff(ndata_smoothed(:,2),sHz); % differentiate y

% plot 1D velocity data
if PlotIt == 1
p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,3),'r'); % plot x-position
% p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,4),'r'); % plot y-position
end

% plot 1D acceleration data
if PlotIt == 1
p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,3),'r'); % plot x-acceleration
% p = plot((1:length(ndata_smoothed))*.005, ndata_smoothed(:,4),'r'); % plot y-acceleration
end


% plot 2D motion data
if PlotIt == 1
p = plot(ndata_smoothed(:,1),ndata_smoothed(:,2),'r.'); % position
% p = plot(ndata_smoothed(:,1),ndata_smoothed(:,2),'r-'); % position scribble
% p = plot(ndata_smoothed(:,3),ndata_smoothed(:,4),'.')% velocity
% p = plot(ndata_smoothed(:,5),ndata_smoothed(:,6),'.') % acceleration
end




% build 2D histogram - manually determined histogram x, y ranges 
nbins = 40;
zzz = histcount2(ndata_smoothed(:,1:2), nbins, [-.3 -.25], [.3 .1]); %2D histogram position
% zzz = histcount2(ndata_smoothed(:,3:4), nbins, [-.3 -.25], [.3 .1]); %2D histogram velocity
% zzz = histcount2(ndata_smoothed(:,5:6), nbins, [-.3 -.25], [.3 .1]); %2D histogram acceleration


% create heatmap
figure
if PlotIt == 1
figure
% s = surf(zzz.X,zzz.Y,(zzz.Z/sum(zzz.Z(:)))); % probability distribution
s = surf(zzz.X,zzz.Y,zzz.Z); % histogram counts
view(2)
hold on
set(s,'LineStyle','none')
set(gca,'clim',[0 2000])
end
