%% PLS R
% nbins = 40;
% day = 1;
% color = {'k','r','g','b'};
% load(['C:\Users\Zack Wright\Documents\MATLAB\DF and FE analysis\ANALYSIS - distribution variables\DF_hists_' num2str(nbins) 'bins_p1top99StrokeRange2_NOSPEED_v4_day' num2str(day) 'PLSR.mat'])
% red_DesignMat(:,1) = [];
% load(['C:\Users\Zack Wright\Documents\MATLAB\DF and FE analysis\ANALYSIS - distribution variables\DF_y_Outcomes_day' num2str(day) '.mat'])
% Metric = 'y_FM';
% yMetric = y_FM;
% 
% N = 22;
% subjects = 1:22;

for model = 1:1
for domain = 0:12
if domain == 0
    bins  = 1:nbins*nbins*9;
elseif domain == 1 || domain == 2 || domain == 3
    bins  = 3*(domain-1)*nbins*nbins+1:nbins*nbins*(domain)*3;
else 
    bins  = (domain-4)*nbins*nbins+1:nbins*nbins*(domain-3);
end


Xin = log(red_DesignMat(:,bins));
Xin(isinf(Xin)) = 0;
% Xin = red_DesignMat(:,bins);

X = zscore(Xin); %red_DesignMat(:,bins)-mean(red_DesignMat(:,bins))
[y, ym, ys] = zscore(yMetric);

% [PCALoadings,PCAScores,PCAVar] = pca(X);
% [beta, fitinfo] = lasso(PCAScores(:,1:21),yMetric);
% lambdas = 0:fitinfo.Lambda(end)./1000:fitinfo.Lambda(end);

PLOTIT=0;
nfolds = 21;
subjects = 1:22;
% whoop = xlsread('whoop_11folds.csv');


% PCAregression
% [PCALoadings,PCAScores,PCAVar] = pca(X);

for ncomp = 1:15
%     ncomp
%     for nrep = 1
%         rng(nrep*100)
% testsubjs = randi(length(subjects),[1 5]);
rep = 1;

yvalfit = zeros(length(subjects),1);
for LO = 1:N
testsubjs = find(subjects == LO);
trainsubjs = find(subjects ~= LO);
Xtrain = X(trainsubjs,:); 
ytrain = y(trainsubjs);
ytrain_NS = (ytrain*ys)+ym;
yActual(LO) = mean(ytrain_NS);%ytrain_NS

if model == 1
% PSLregression
% CV = cvpartition(length(trainsubjs),'leaveout');
% [XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(Xtrain,ytrain,ncomp,'cv',CV);
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(Xtrain,ytrain,ncomp);
elseif model == 2
% PCAregression
[PCALoadings,PCAScores,PCAVar] = pca(Xtrain);
beta = regress(ytrain-mean(ytrain), PCAScores(:,1:ncomp));
beta = PCALoadings(:,1:ncomp)*beta;
beta = [mean(ytrain) - mean(Xtrain)*beta; beta];
% yfitPCR = [ones(n,1) Xtrain]*betaPCR;
% PCRmsep = sum(crossval(@pcrsse,X,y,'KFold',10),1) / n;
end

train_yfit = [ones(size(Xtrain,1),1) Xtrain]*beta;
train_yfit = (train_yfit*ys)+ ym;
TSS = sum((ytrain_NS-mean(ytrain_NS)).^2);
RSS = sum((ytrain_NS-train_yfit).^2);
train_R2(ncomp,LO,domain+1) = 1 - RSS/TSS;
train_RMSE(ncomp,LO,domain+1) =  sqrt(mean(abs(ytrain_NS-train_yfit).^2));
train_MSE(ncomp,LO,domain+1) =  mean(abs(ytrain_NS-train_yfit).^2);


Xtest = X(testsubjs,:); 
% ytest = y(testsubjs);
yvalfit(testsubjs,1) = [ones(size(Xtest,1),1) Xtest]*beta;

rep = rep+1;
end



if PLOTIT==1
figure
plot(1:ncomp,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

% figure
% plot(ytrain,yfit,'o')
end

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
test_SE_ns(ncomp,:,domain+1) = (y-yvalfit).^2; %Check
  yvalfit = (yvalfit*ys)+ ym;
  test_P(:,ncomp,domain+1) = yvalfit;
  test_PRESS(ncomp,domain+1) = sum(abs(yMetric-yvalfit).^2);
  TSSVal = sum(abs(yMetric-yActual').^2);
  test_PRESSR2(ncomp,domain+1) = 1-(test_PRESS(ncomp,domain+1)/TSSVal);
  test_RMPRESS(ncomp,domain+1) =  sqrt(mean(abs(yMetric-yvalfit).^2));
  test_SE(ncomp,:,domain+1) =  (yMetric-yvalfit).^2;
  


 if model == 1 
  % PSLregression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X,y,ncomp);
 else
% PCAregression
[PCALoadings,PCAScores,PCAVar] = pca(Xtrain);
beta = regress(ytrain-mean(ytrain), PCAScores(:,1:ncomp));
beta = PCALoadings(:,1:ncomp)*beta;
beta = [mean(ytrain) - mean(Xtrain)*beta; beta];
% % yfitPCR = [ones(n,1) Xtrain]*betaPCR;
% % PCRmsep = sum(crossval(@pcrsse,X,y,'KFold',10),1) / n;
 end
 
full_yfit = [ones(size(X,1),1) X]*beta;
yfull_NS = (y*ys)+ym;
full_yfit = (full_yfit*ys)+ ym;
TSS = sum((yfull_NS-mean(yfull_NS)).^2);
RSS = sum((yfull_NS-full_yfit).^2);
full_R2(ncomp,domain+1) = 1 - RSS/TSS;
full_RMSE(ncomp,domain+1) =  sqrt(mean(abs(yfull_NS-full_yfit).^2));
full_MSE(ncomp,domain+1) =  mean(abs(yfull_NS-full_yfit).^2);
  
% close all
%     end
end
end

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
set(gca,'xlim',[0 11])
xlabel('number of components')
ylabel('Mean Squared Error')
 
% figure
% hold on
% for domain = 1:4
% for a = 1:ncomp
% %     plot1DDistributionV2(test_PRESSR2(a,:),'b',[0.9,1.1]+a-1,[.5 .5 .5],200)
%         plot(a+.1*domain,test_PRESSR2(a,domain),'color',color{domain},'marker','.')
% %         plot(a+.1*domain,mean(train_R2(a,:,domain)'),'color',color{domain},'marker','o')
%     wing=std(train_R2(a,:,domain))/sqrt(length(test_SE(a,:,domain))); 
%     errorbar(a+.1*domain,mean(train_R2(a,:,domain)'),wing,'color',color{domain},'LineStyle','none','LineWidth',1,'marker','.','markersize',16,'MarkerFaceColor', 'none')
%     plot(a+.1*domain,mean(full_R2(a,domain)'),'color',color{domain},'marker','o')
% 
% end
% end


% figure
% hold on
% for domain = 1:4
% for a = 1:ncomp
% %     plot1DDistributionV2(test_PRESSR2(a,:),'b',[0.9,1.1]+a-1,[.5 .5 .5],200)
%     plot1DDistributionV2(train_RMSE(a,:,domain),color{domain},[a-.1,a+.1]+.1*domain,[.5 .5 .5],200)
% %     plot(a,train_RMSE(a,domain),color{domian})
% end
% end


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


figure
hold on
scf = 100; % scale factor for x-scatter

for domain = 0:3
for a = 1:5
plot1DDistributionV2(test_SE(a,:,domain+1),color{domain+1},[-.1,.1]+domain*.1+a,[.5 .5 .5],scf,'.')
% plot1DDistributionV2(train_MSE(a,:,domain+1),color{domain+1},[-.1,.1]+domain*.1+a+.05,[.5 .5 .5],scf,'o')
 plot1DDistributionV2(full_MSE(a,domain+1),color{domain+1},[-.1,.1]+domain*.1+a+.05,[.5 .5 .5],scf,'o')



end
end
ylabel('Mean Squared Error')
xlabel('')
set(gcf,'color','w','units','inches','Position',[2 2 3 3])
set(gca,'xlim',[.75 5.5],'XTickLabel',[],'Fontsize',10)
set(gca,'Position',[.175 .15 .8 .8])






for domain = 0:12
%     [~,ncompmin2(domain+1)]  = min(mean(test_SE(:,:,domain+1),2));
    ncompmin2(domain+1)  = 1;

end

if model == 1
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
else
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
end

% writetable(TPSLR,['TPLSR_nbins' num2str(nbins) '_day' num2str(day) '_' Metric '1ncomp.csv'],'WriteRowNames',true)
% writetable(TPCAR,['TPCAR_nbins' num2str(nbins) '_day' num2str(day) '_' Metric '1ncomp.csv'],'WriteRowNames',true)

