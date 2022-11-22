function [outline_coords, outline_area] = get_outlineXY(x,y,x0,y0,perc,div,max_theta)
%Inputs: x,y - state space variables (n x 1)
%        x0, y0 - point in x,y space to calculate coverage (1x1)
%        perc -  percent coverage
%        div - divide the space by dtheta div number of times
%        max_theta - 
% Outputs: outline_coods - the x,y corrdinates for the x% coverage outline
%          outline_area - The estimated x% coverage
% Example:
%     [o, o_area] = get_outlineXY(x,y,0,0,90,64,2*pi);


% x = MasterD.FE{2}.phase{phases(1)}.cumulative(1:2:end,1);
% y = MasterD.FE{2}.phase{phases(1)}.cumulative(1:2:end,2);
% x = -x;
% y = -y+.7;
% plot(x,y,'.')
% hold on
% x0 = 0; y0 = 0;
% r = 1;
% d20 = sqrt(x.^2 + y.^2);
% thetas = acos(dot([x y]',[ones(length(x),1) zeros(length(x),1)]')'./(d20.*r));
% perc = 90;
% drange=prctile(d20,perc);
% for angle = 0:pi/128:pi-(pi/128)
%     ind = find(thetas(:) >= angle & thetas(:) < angle+(pi/128));
%     [m,i] = max(d20(ind));
%     plot(x(ind(i)),y(ind(i)),'m*')
% 
%     halfwayangle = angle+((pi/128)/2);
%     drange=prctile(d20(ind), perc); 
%     xnew = drange*cos(halfwayangle);
%     ynew = drange*sin(halfwayangle);
%     plot(xnew,ynew,'k*')
% 
%     [m,i] = min(abs(drange-d20(ind)));
%     plot(x(ind(i)),y(ind(i)),'g*')
% 
%     plot([0 1.5*cos(angle)],[0 1.5*sin(angle)],'r')
% end




% figure
% x = MasterD.FE{2}.phase{phases(1)}.cumulative(:,3);
% y = MasterD.FE{2}.phase{phases(1)}.cumulative(:,4);
% x = -x;
% y = -y;
% plot(x,y,'o')
% hold on
% x0 = 0; y0 = 0;
x = x-x0;
y = y-y0;
r = 1;
% xC = r*cos(angles);
% yC = r*sin(angles);
d20 = sqrt(x.^2 + y.^2);
thetas = acos(dot([x y]',[ones(length(x),1) zeros(length(x),1)]')'./(d20.*r));
thetas = atan2(x.*zeros(length(x),1)-ones(length(x),1).*y,x.*ones(length(x),1)+y.*zeros(length(x),1));
thetas = mod(-thetas,2*pi);

% perc = 90;
% drange2=prctile(d20,perc);
% div = 128;
xnew = [];
ynew = [];
sumind = 0;
for angle = 0:pi/div:max_theta - (pi/div)
    ind = find(thetas(:) >= angle & thetas(:) < angle+(pi/div));
%     [m,i] = max(d20(ind));
%     plot(x(ind(i)),y(ind(i)),'m*')
drange=prctile(d20(ind), perc); 
    halfwayangle = angle+((pi/div)/2);
    xnew = [xnew;drange*cos(halfwayangle)];
    ynew = [ynew;drange*sin(halfwayangle)];
%     plot(xnew,ynew,'k*')
    

    
%     [m,i] = min(sqrt((x(ind)-xnew).^2 + (y(ind)-ynew).^2));% min(abs(drange-d20(ind)))<-This one is minumum distance to the origin
%     plot(x(ind(i)),y(ind(i)),'g*')
% 
%     plot([0 1.5*cos(angle)],[0 1.5*sin(angle)],'r')
    
    
end

outline_coords = [xnew ynew];
outline_area = sum(AreaTriangle([ones(length(outline_coords))*x0 ones(length(outline_coords))*y0], outline_coords, [outline_coords(2:end,:);outline_coords(1,:)]));

outline_coords(end+1,:) = [outline_coords(1,1) outline_coords(1,2)];  % to connect outline for plotting
outline_coords(:,1) = outline_coords(:,1)+x0;
outline_coords(:,2) = outline_coords(:,2)+y0;

    
    
