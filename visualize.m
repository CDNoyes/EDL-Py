%% Baseline viewing
base = load('C:\Users\cdnoyes\Documents\Experimental\Python\data\Baseline.mat');

figure()
plot(base.states(:,2),base.states(:,3),'k','LineWidth',3)

figure()
plot(base.states(:,4),(base.states(:,1)-3397e3)/1000,'k','LineWidth',3)




%%
clear
% addpath('C:\Users\cdnoyes\Documents\Experimental\Matlab')
dtr = pi/180;
base = load('C:\Users\cdnoyes\Documents\Experimental\Python\data\Baseline.mat');
mc =  load('C:\Users\cdnoyes\Documents\Experimental\Python\data\MC.mat');
pce = load('C:\Users\cdnoyes\Documents\Experimental\Python\data\PCE2.mat');

ind = base.index;
base = rmfield(base,'index');
mc.mean = squeeze(mean(mc.states,1));
data = [mc,pce];
label = {'MC','PCE'};

for d = 1:length(data)
    states = data(d).states;
    samples = data(d).samples;
    figure
    title(label{d})
    plot(base.states(:,2),base.states(:,3),'k','LineWidth',3)
    hold all
    for i = 1:size(states,1)
        plot(states(i,:,2),states(i,:,3))
    end
    
    figure
    title(label{d})
    plot(base.states(:,4),(base.states(:,1)-3397e3)/1000,'k','LineWidth',3)
    % ParachuteDeploymentConstraints(true);
    hold all
    for i = 1:size(states,1)
        plot(states(i,:,4),(states(i,:,1)-3397e3)/1000,'b')
    end
    %     plot(pce.mean(:,4),(pce.mean(:,1)-3397e3)/1000,'r','LineWidth',4)
    
    figure
    plot(states(:,end,4),(states(:,end,1)-3397e3)/1000,'o')
    title(label{d})
    
end
%% Comparison between PCE and MC results:
err = data(1).states-data(2).states;
errSamp = data(1).samples-data(2).samples;
errMean = abs(data(1).mean-data(2).mean);
nonlin = base.states-data(1).mean;
for i = 1:size(errMean,2)
    for j = 1:size(err,2)
        errNorm(j,i) = norm(err(:,j,i));
    end
end
en = linspace(0,1,size(errMean,1));
figure
subplot(2,2,1)
histogram(err(:,end,4),'Normalization','pdf','BinMethod','scott')
title('Error between MC and 2nd Order PCE Approximation')
subplot(2,2,3)
plot(err(:,end,4),(err(:,end,1))/1000,'o')
xlabel('Velocity Error (m/s)')
ylabel('Altitude Error (km)')
subplot(2,2,4)
histogram(err(:,end,1)/1000,'Normalization','pdf','BinMethod','scott')

figure
subplot(2,2,1)
histogram(err(:,end,2)/dtr,'Normalization','pdf','BinMethod','scott')
title('Error between MC and 2nd Order PCE Approximation')
subplot(2,2,3)
plot(err(:,end,2)/dtr,(err(:,end,3))/dtr,'o')
xlabel('Lon Error (deg)')
ylabel('Lat Error (deg)')
subplot(2,2,4)
histogram(err(:,end,3)/dtr,'Normalization','pdf','BinMethod','scott')


figure
plot(en,errMean(:,1)/1000)
% hold all
% plot(en,errNorm(:,1)/1000)
% legend('Means','Norms')
ylabel('Error ')
xlabel('Normalized Energy')
figure
plot(en,errMean(:,4))
% hold all
% plot(en,errNorm(:,4))
% legend('Means','Norms')
xlabel('Normalized Energy')
ylabel('Error between mean velocity estimations')


for i = [2,3,5,6]
    figure
    plot(en,errMean(:,i)/dtr)
    xlabel('Normalized Energy')
    ylabel(['Error between mean state ',num2str(i),' estimations (deg)'])
    
end

scales([2,3,5,6]) = dtr;
scales(1) = 1000;
scales(4) = 1;
figure
for i = 1:6
    figure
    plot(en,nonlin(:,i)/scales(i))
    xlabel('Normalized Energy')
    ylabel(['Difference between mean estimation and baseline state ',num2str(i),''])
    
end
% for i = 1:size(samples,1)
%     figure
%     hist(samples(i,:))
%
% end