function y=histcount2(DDD0, nbins, drange1, drange2)

DDD=DDD0(:,[1 2]);

d01=drange1(1:2);
d99=drange2(1:2);

DDDr=repmat(d99-d01,length(DDD),1);   %Matrix of ranges
DDDm=repmat(d01, length(DDD),1);           %Matrix of min


DDD=round((nbins-1)*(DDD-DDDm)./DDDr)+1;        %Normalized, offset data


DDD=min(DDD, nbins*ones(size(DDD)));
DDD=max(DDD, ones(size(DDD)));


ZCNT3=zeros(nbins, nbins);    %Empty Count Matrix per frame

for a = 1:nbins
    for b = 1:nbins
Zind{a,b}=[];
    end
end


for kk=1:length(DDD);
    
    ZCNT3(DDD(kk,1),DDD(kk,2))=...
        ZCNT3(DDD(kk,1),DDD(kk,2))+1; %Add count to current address
    
%     Zind{DDD(kk,1),DDD(kk,2)}
%     DDD(kk,1)
%     DDD(kk,2)
    Zind{DDD(kk,1),DDD(kk,2)} = [Zind{DDD(kk,1),DDD(kk,2)},kk];
end

% [DDD(1:end-1,1) DDD(1:end-1,2) diff(DDD(:,1)) diff(DDD(:,2))]

[XX,YY]=ndgrid(1:nbins, 1:nbins);
y.X=((d99(1)-d01(1))*(XX-1))/(nbins-1)+d01(1);        %Normalized, offset data
y.Y=((d99(2)-d01(2))*(YY-1))/(nbins-1)+d01(2);        %Normalized, offset data

y.Z=ZCNT3;


y.Zind = Zind;
