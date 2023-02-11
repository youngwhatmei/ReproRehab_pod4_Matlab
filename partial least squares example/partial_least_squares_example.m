%% partial_least_squares_example.m
% PCA and PLS Regression analysis on 2D histogram features built from free 
% exploration data across 9 movement variables to predict upper limb impairment level in stroke survivors. 

%% Load histogram feature data and Subject Outcome data (see README.txt)
nbins = 40;
day = 1;
color = {'k','r','g','b'}; % colors for movement variable domains ('k' - all domains combined, 'r' - endpoint kinematics, 'g' - joint kinematics, 'b' - joint kinetics)  
load(['\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\partial least squares example\DF_hists_' num2str(nbins) 'bins_p1top99StrokeRange2_NOSPEED_v4_day' num2str(day) 'PLSR.mat'])
red_DesignMat(:,1) = [];
load(['\\wsl$\ubuntu\home\zwright\ReproRehab_pod4_Matlab\partial least squares example\DF_y_Outcomes_day' num2str(day) '.mat'])
Metric = 'y_FM';
yMetric = y_FM;

%% static and initialized variables
N = 22; % number of subjects
subjects = 1:22;
PLOTIT=0; % plot if = 1
nfolds = 21; % choose number of folds for cross-validation - N-1 is leave-one out cross-validation 
max_ncomp = 5; % specifiy number of pca or pls components to include in the regression analysis (max is N-1)
domainorder = 1:13;
model = 1; % model 1 - PLS regression, model 2 - PCA regression

%% Run PCA and PLS regression analysis
% runs analysis for 1) all movement variables combined, 2) each movement variable domain and 3) each individual movement variable
for domain = 0:12 % run regression analysis using specified movement variable domains (each domain includes 3 movement vairables)
if domain == 0
    bins  = 1:nbins*nbins*9; % include all features across all movement variables
elseif domain == 1 || domain == 2 || domain == 3 % 
    bins  = 3*(domain-1)*nbins*nbins+1:nbins*nbins*(domain)*3; % include features for specified domain of movement variables (1 - endpoint kinematics, 2 - joint kinematics, 3 - joint kinetics)  
else 
    bins  = (domain-4)*nbins*nbins+1:nbins*nbins*(domain-3); % include features for a single movement variable
end

% Xin = red_DesignMat(:,bins); % raw histogram count feature set
Xin = log(red_DesignMat(:,bins)); % (alternate feature set) log of the feature set
Xin(isinf(Xin)) = 0; % remove inf if exists in feature set

% standardize feature set
X = zscore(Xin); %red_DesignMat(:,bins)-mean(red_DesignMat(:,bins))
[y, ym, ys] = zscore(yMetric);

for ncomp = 1:max_ncomp % run analysis for successive number of components

yvalfit = zeros(length(subjects),1);
for LO = 1:N % run analysis for each leave-one(subject)-out cross-validataion
testsubjs = find(subjects == LO);
trainsubjs = find(subjects ~= LO);
Xtrain = X(trainsubjs,:); 
ytrain = y(trainsubjs);
ytrain_NS = (ytrain*ys)+ym;
yActual(LO) = mean(ytrain_NS);

if model == 1 % PSL regression
% (option 1 - cross-validataion leaving out more than one subject) 
% CV = cvpartition(length(trainsubjs),'leaveout');
% [XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(Xtrain,ytrain,ncomp,'cv',CV);

% (option 2 - leave-one(subject)-out cross-validataion)
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(Xtrain,ytrain,ncomp);

elseif model == 2 % PCA regression
% (option 2 - leave-one(subject)-out cross-validataion)
[PCALoadings,PCAScores,PCAVar] = pca(Xtrain);
beta = regress(ytrain-mean(ytrain), PCAScores(:,1:ncomp));
beta = PCALoadings(:,1:ncomp)*beta;
beta = [mean(ytrain) - mean(Xtrain)*beta; beta];
% yfitPCR = [ones(n,1) Xtrain]*betaPCR;
% PCRmsep = sum(crossval(@pcrsse,X,y,'KFold',10),1) / n;
end

% fit regression model to training data and evaluate error 
train_yfit = [ones(size(Xtrain,1),1) Xtrain]*beta;
train_yfit = (train_yfit*ys)+ ym;
TSS = sum((ytrain_NS-mean(ytrain_NS)).^2);
RSS = sum((ytrain_NS-train_yfit).^2);
train_R2(ncomp,LO,domain+1) = 1 - RSS/TSS; % coefficient of determination
train_RMSE(ncomp,LO,domain+1) =  sqrt(mean(abs(ytrain_NS-train_yfit).^2)); % root mean squared error
train_MSE(ncomp,LO,domain+1) =  mean(abs(ytrain_NS-train_yfit).^2); % mean squared error

Xtest = X(testsubjs,:); 
yvalfit(testsubjs,1) = [ones(size(Xtest,1),1) Xtest]*beta;
end


% CHECK = plot Percent Variance Explained (for PLS only)
if PLOTIT==1
figure
hold on
plot(1:ncomp,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
% figure
% plot(ytrain,yfit,'o')
end

% CHECK = plot MSE (for PLS only)
if PLOTIT==1
% figure
% plot(1:ncomp,stats.W,'o-');
% legend({'c1','c2','c3','c4','c5','c6'},'Location','NW')
% xlabel('Predictor');
% ylabel('Weight');
figure
[axes,h1,h2] = plotyy(0:ncomp,MSE(1,:),0:ncomp,MSE(2,:));
set(h1,'Marker','o')
set(h2,'Marker','o')
legend('MSE Predictors','MSE Response')
xlabel('Number of Components')
end

% fit regression model to test data and evaluate error 
  test_SE_ns(ncomp,:,domain+1) = (y-yvalfit).^2; %Check
  yvalfit = (yvalfit*ys)+ ym;
  test_P(:,ncomp,domain+1) = yvalfit;
  test_PRESS(ncomp,domain+1) = sum(abs(yMetric-yvalfit).^2);
  TSSVal = sum(abs(yMetric-yActual').^2);
  test_PRESSR2(ncomp,domain+1) = 1-(test_PRESS(ncomp,domain+1)/TSSVal);
  test_RMPRESS(ncomp,domain+1) =  sqrt(mean(abs(yMetric-yvalfit).^2));
  test_SE(ncomp,:,domain+1) = (yMetric-yvalfit).^2;
  

% Full Model - regression analysis on the all subjects (no cross validation
 if model == 1 % PLS regression
  % PSLregression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X,y,ncomp);
 elseif model == 2 % PCA regression
% PCAregression
[PCALoadings,PCAScores,PCAVar] = pca(Xtrain);
beta = regress(ytrain-mean(ytrain), PCAScores(:,1:ncomp));
beta = PCALoadings(:,1:ncomp)*beta;
beta = [mean(ytrain) - mean(Xtrain)*beta; beta];
% % yfitPCR = [ones(n,1) Xtrain]*betaPCR;
% % PCRmsep = sum(crossval(@pcrsse,X,y,'KFold',10),1) / n;
 end

% fit regression Full Model and evaluate error  
full_yfit = [ones(size(X,1),1) X]*beta;
yfull_NS = (y*ys)+ym;
full_yfit = (full_yfit*ys)+ ym;
TSS = sum((yfull_NS-mean(yfull_NS)).^2);
RSS = sum((yfull_NS-full_yfit).^2);
full_R2(ncomp,domain+1) = 1 - RSS/TSS;
full_RMSE(ncomp,domain+1) =  sqrt(mean(abs(yfull_NS-full_yfit).^2));
full_MSE(ncomp,domain+1) =  mean(abs(yfull_NS-full_yfit).^2);
  
end
end

%% plot mean squared error for test and train data for successive number of components
figure
hold on
for domain = 1:4
for a = 1:ncomp
%     plot1DDistributionV2(test_SE(a,:,domain),color{domain},[a-.1,a+.1]+.1*domain,[.5 .5 .5],200,'.')
%     plot(a,test_PRESSR2(a,:),'ro')
%     plot1DDistributionV2(train_MSE(a,:,domain),color{domain},[a-.1,a+.1]+.1*domain,[.5 .5 .5],200,'o')
%     wing=confidence(test_SE(a,:,domain),.95); 
    wing=std(test_SE(a,:,domain))/sqrt(length(test_SE(a,:,domain))); 
    errorbar(a+.1*domain,mean(test_SE(a,:,domain)'),wing,'color',color{domain},'LineStyle','none','LineWidth',1,'marker','.','markersize',16)
%     wing=confidence(train_MSE(a,:,domain),.95); 
    wing=std(train_MSE(a,:,domain))/sqrt(length(test_SE(a,:,domain))); 
    errorbar(a+.1*domain,mean(train_MSE(a,:,domain)'),wing,'color',color{domain},'LineStyle','none','LineWidth',1,'marker','.','markersize',16,'MarkerFaceColor', 'none')
    plot(a+.1*domain,mean(full_MSE(a,domain)'),'color',color{domain})
end
end
set(gcf,'color','w','units','inches','Position',[2 2 3 3])
set(gca,'xlim',[0 max_ncomp+1])
xlabel('number of components')
ylabel('Mean Squared Error')

%% plot coefficient of determination for test and train data for successive number of components
figure
hold on
for domain = 1:4
for a = 1:ncomp
%   plot1DDistributionV2(test_PRESSR2(a,:),'b',[0.9,1.1]+a-1,[.5 .5 .5],200)
    plot(a+.1*domain,test_PRESSR2(a,domain),'color',color{domain},'marker','.')
%   plot(a+.1*domain,mean(train_R2(a,:,domain)'),'color',color{domain},'marker','o')
    wing=std(train_R2(a,:,domain))/sqrt(length(test_SE(a,:,domain))); 
    errorbar(a+.1*domain,mean(train_R2(a,:,domain)'),wing,'color',color{domain},'LineStyle','none','LineWidth',1,'marker','.','markersize',16,'MarkerFaceColor', 'none')
    plot(a+.1*domain,mean(full_R2(a,domain)'),'color',color{domain},'marker','o')
end
end
set(gcf,'color','w','units','inches','Position',[2 2 3 3])
set(gca,'xlim',[0 max_ncomp+1])
xlabel('number of components')
ylabel('Coefficient of Determination')

%% plot actual vs. predicted y outcome
figure
hold on
ncomp = 1;
line([0 50],[mean(yMetric) mean(yMetric)],'color','k','LineStyle','--')
for domain = 0:3
plot(yMetric, test_P(:,ncomp,domain+1),'color',color{domain+1},'LineStyle','none','Marker','.','MarkerSize',12)
end
lsline
ylabel('Predicted Fugl-Meyer Score')
xlabel('Actual Fugl-Meyer Score')
set(gca,'xlim',[15 50],'ylim',[15 50],'XTick',[15:5:50],'YTick',[15:5:50],'Fontsize',10)
set(gcf,'color','w','units','inches','Position',[2 2 3 3])
set(gca,'Position',[.15 .15 .8 .8])

%% Organize results into a table and Write to a CSV
for domain = 0:12
%     [~,ncompmin2(domain+1)]  = min(mean(test_SE(:,:,domain+1),2));
    ncompmin2(domain+1)  = 1;

end

if model == 1 % PLS regression
    for domain = 1:13
        tp1(domain) = test_PRESSR2(ncompmin2(domainorder(domain)),domainorder(domain));
        tp2(domain) = squeeze(mean(train_R2(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp3(domain) = squeeze(std(train_R2(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp4(domain) = full_R2(ncompmin2(domainorder(domain)),domainorder(domain));
        tp5(domain) = squeeze(mean(test_SE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp6(domain) = squeeze(std(test_SE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp7(domain) = squeeze(mean(train_MSE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp8(domain) = squeeze(std(train_MSE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp9(domain) = full_MSE(ncompmin2(domainorder(domain)),domainorder(domain));
    end
% Results = [test_PRESSR2(ncomp,:)', mean(train_R2)', mean(test_SE)', mean(train_MSE)'];
TPSLR = table(ncompmin2(domainorder)',tp1', tp2', tp3', tp4', tp5', tp6', tp7',tp8',tp9',...
    'RowNames',{'Full','EP','x pos','x vel','x acc','Joint','q pos','q vel','q acc','Kinetic','force','torque','power'},... 
    'VariableNames',{'ncomp','testR2','trainR2','SDtrainR2','AllR2','testMSE','SDtestMSE','trainMSE','SDtrainMSE','AllMSE'}); 
elseif model == 2 % PCA regression
    for domain = 1:13
        tp1(domain) = test_PRESSR2(ncompmin2(domainorder(domain)),domainorder(domain));
        tp2(domain) = squeeze(mean(train_R2(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp3(domain) = squeeze(std(train_R2(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp4(domain) = full_R2(ncompmin2(domainorder(domain)),domainorder(domain));
        tp5(domain) = squeeze(mean(test_SE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp6(domain) = squeeze(std(test_SE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp7(domain) = squeeze(mean(train_MSE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp8(domain) = squeeze(std(train_MSE(ncompmin2(domainorder(domain)),:,domainorder(domain))));
        tp9(domain) = full_MSE(ncompmin2(domainorder(domain)),domainorder(domain));
    end
% Results = [test_PRESSR2(ncomp,:)', mean(train_R2)', mean(test_SE)', mean(train_MSE)'];
TPCAR = table(ncompmin2(domainorder)',tp1', tp2', tp3', tp4', tp5', tp6', tp7',tp8',tp9',...
    'RowNames',{'Full','EP','x pos','x vel','x acc','Joint','q pos','q vel','q acc','Kinetic','force','torque','power'},... 
    'VariableNames',{'ncomp','testR2','trainR2','SDtrainR2','AllR2','testMSE','SDtestMSE','trainMSE','SDtrainMSE','AllMSE'}); 
end

% writetable(TPSLR,['TPLSR_nbins' num2str(nbins) '_day' num2str(day) '_' Metric '1ncomp.csv'],'WriteRowNames',true)
% writetable(TPCAR,['TPCAR_nbins' num2str(nbins) '_day' num2str(day) '_' Metric '1ncomp.csv'],'WriteRowNames',true)

