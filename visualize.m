%%
clc;clear
dtr = pi/180;
data_path = 'E:\Documents\EDL\data\'; % Desktop
% data_path = 'C:\Users\cdnoyes\Documents\EDL\data\'; % Laptop

% data(1) = load([data_path, 'MC_NMPC_2000_AltitudeTrigger.mat']);
% data(1) = load([data_path, 'MC_NMPC_2000_TimeTrigger_NoLateral.mat']);
% data(2) = load([data_path, 'MC_NMPC_2000_TimeTrigger_NoLateral_tanh.mat']);
data(1) = load([data_path, 'MC_NMPC_2000.mat']);
data(2) = load([data_path, 'MC_NMPC_2000_EnergyTrigger.mat']);

% data(2) = load([data_path, 'MC_NMPC_2000_EnergyTrigger_NoLateral.mat']);


% data(2) = load([data_path, 'MC_NMPC_2000_NoLateral_NoMargin.mat']);

% data(2) = load([data_path, 'MC_NMPC_2000_TimeTrigger.mat']);
% data(3) = load([data_path, 'MC_NMPC_2000_EnergyTrigger.mat']);

% data(2) = load([data_path, 'MC_NMPC_2000.mat']);
% data(3) = load([data_path, 'MC_NMPC_300_update1.mat']);

targetDR = 880;

itime = 1;
idrag = 14;
ienergy = 2;
irange = 11;
ialt = 5; % 5 is actually radius, 4 is alt in km 
ivel = 8;
ifpa = 9;
ibank = 3;
final = zeros(length(data),length(data(1).pdf),27);
% label = {'Alt. Rate','Drag Rate','No Rate'};
% label = {'Open Loop','CTNPC','CTNPC+Update1'};
label = {'Default','Optimized'};

colors = {'r','b','g','m'};
for d = 1:length(data)
    states = data(d).states;
    samples = data(d).samples;
    %     pdf(d,:) = data(d).pdf;
    figure(2+d)
    hold on
    for i = 1:length(states)
        if 1
            plot(states{i}(:,irange),states{i}(:,ialt)/1000-3397)
%             plot(states(i,:,ivel),(states(i,:,ialt-1)))

        else
            plot(states{i}(:,itime),(states{i}(:,ibank)))
%             plot(states(i,:,itime),(states(i,:,ibank)))

        end
        title(label{d})
        final(d,i,:) = states{i}(end,:);
%         final(d,i,:) = states(i,end,:);

        
    end
    
    
end

% final(3,1:300,12) = final(3,1:300,12) + 3;
% final(3,1:300,11) = targetDR-0.5 + -0.1 + 0.2*randn(1,300);
for d = 1:length(data)
    
    figure(1)
    hold all
    plot(final(d,:,8),(final(d,:,5)-3397e3)/1000, 'o')
    xlabel('Velocity (m/s)')
    ylabel('Altitude (km)')
    
    figure(2)
    hold all
    plot(final(d,:,12),(final(d,:,11)), 'o')
    xlabel('CR (km)')
    ylabel('DR (km)')
end
legend(label{:})
figure(1)
legend(label{:})
% Ellipse([0,targetDR],2,2,0,{'r--','LineWidth',2})
% Ellipse([0,targetDR],5,5,0,{'k--','LineWidth',2})
% Ellipse([0,targetDR],10,10,0,'b--')

for i =1:length(data)
    len = length(data(i).states);
    dist = sqrt(sum(0*final(i,1:len,12).^2 +(final(i,1:len,11)-targetDR).^2,1));
    disp(['Mean miss distance = ', num2str(mean(dist)),' km'])
    % disp(['Std dev miss distance ', num2str(std(dist)),' km'])
    disp(['standard deviation miss distance = ', num2str(std(dist)),' km'])
    disp(' ')
    metric = prctile(final(i,1:len,6),99)-prctile(final(i,1:len,6),1) + ...
            prctile(final(i,1:len,7),99)-prctile(final(i,1:len,7),1);
    disp(['Miss metric 99%-1% = ', num2str(metric)])
    
    ntotal = length(dist);
    n2 = length(dist(dist>2));
    n5 = length(dist(dist>5));
    n10 = length(dist(dist>10));
    p2 = 1-n2/ntotal;
    p5 = 1-n5/ntotal;
    p10 = 1-n10/ntotal;
    h = (final(i,1:len,5)-3397e3)/1000;
    n_min_alt = length(h(h < 0.52));
    p_min_alt = n_min_alt/ntotal;
    disp(['Mean altitude ', num2str(mean(h)),' km'])
    disp(['1%-ile altitude ', num2str(prctile(h,1)),' km'])


    figure
    hold on
%     Ellipse([0,targetDR],2,2,0,{'r--','LineWidth',2})
%     Ellipse([0,targetDR],5,5,0,{'k--','LineWidth',2})
%     Ellipse([0,targetDR],10,10,0,'b--')
    
    plot(final(i,1:len,12),final(i,1:len,11),'o')%,pdf(i,:)/max(pdf(i,:)))
    legend([' 2 km (',num2str(p2*100),'% inside)'], [' 5 km (',num2str(p5*100),'% inside)'], ['10 km (',num2str(p10*100),'% inside)'])
    xlabel('CR (km)')
    ylabel('DR (km)')
    axis equal
    box on
    grid on
    
    % figure
    % scatter(final(i,:,8),(final(i,:,5)-3397e3)/1000,[],pdf(i,:)/max(pdf(i,:)))
    % xlabel('Velocity (m/s)')
    % ylabel('Altitude (km)')
    % title('Colored by probability')
    % box on
    % grid on
    figure
    scatter(final(i,1:len,8),h,[],dist)
    xlabel('Velocity (m/s)')
    ylabel('Altitude (km)')
    title([label{i}, ', ',num2str(100*p_min_alt), '% on minimum altitude boundary'])
    box on
    grid on
end
% colorbar
% figure
% scatter(samples(1,:),samples(2,:),[],pdf(1,:)/max(pdf(1,:)))
% figure
% scatter(samples(3,:),samples(4,:),[],pdf(1,:)/max(pdf(1,:)))
return

%% Baseline viewing
base = load('E:\Documents\EDL\data\Baseline.mat');

figure()
plot(base.states(:,2),base.states(:,3),'k','LineWidth',3)

figure()
plot(base.states(:,4),(base.states(:,1)-3397e3)/1000,'k','LineWidth',3)




%%
clear
% addpath('C:\Users\cdnoyes\Documents\Experimental\Matlab')
dtr = pi/180;
base = load('E:\Documents\EDL\data\PCE_Comparison\Baseline.mat');
mc =  load('E:\Documents\EDL\data\PCE_Comparison\MC.mat');
pce = load('E:\Documents\EDL\data\PCE_Comparison\PCE2.mat');

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
    
    figure(100)
    %     c = {'b','r'};
    hold all
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
%
% scales([2,3,5,6]) = dtr;
% scales(1) = 1000;
% scales(4) = 1;
% figure
% for i = 1:6
%     figure
%     plot(en,nonlin(:,i)/scales(i))
%     xlabel('Normalized Energy')
%     ylabel(['Difference between mean estimation and baseline state ',num2str(i),''])
%
% end
% for i = 1:size(samples,1)
%     figure
%     hist(samples(i,:))
%
% end

%% Cost function comparison
dtr = pi/180;
base = load('E:\Documents\EDL\data\CostFunction\Baseline.mat');
mc =  load('E:\Documents\EDL\data\CostFunction\MC.mat');
pce = load('E:\Documents\EDL\data\CostFunction\PCE2.mat');
pdf = load('E:\Documents\EDL\data\CostFunction\samplePDF.mat');
ind = base.index;
base = rmfield(base,'index');
mc.mean = squeeze(mean(mc.states,2));
data = [mc,pce];
label = {'MC','PCE'};

err = data(2).states-data(1).states;
errPer = 100*err./data(1).states;
errSamp = data(2).samples-data(1).samples;
errMean = abs(data(1).mean-data(2).mean);

for i=1:length(data)
    figure(1)
    subplot(1,2,i)
    scatter(data(i).samples(1,:)*100,data(i).samples(2,:)*100,[],data(i).states)
    title(label{i})
    xlabel('CD offset (%)')
    ylabel('CL offset (%)')
    
    figure(2)
    subplot(1,2,i)
    scatter(data(i).samples(3,:)*100,data(i).samples(4,:)*100,[],data(i).states)
    title(label{i})
    xlabel('rho0 offset (%)')
    ylabel('scale height offset (%)')
    
    
end

figure(3)
subplot(1,2,1)
scatter(data(1).samples(1,:)*100,data(1).samples(2,:)*100,[],errPer)
title('Error in Cost Function')
xlabel('CD offset (%)')
ylabel('CL offset (%)')
colorbar
subplot(1,2,2)
scatter(data(1).samples(3,:)*100,data(1).samples(4,:)*100,[],errPer)
title('Error in Cost Function')
xlabel('rho0 offset (%)')
ylabel('scale height offset (%)')


figure
plot(pdf.pdf,errPer,'o')
title('Error in PCE Cost Function Approximation ')
xlabel('Sample Density (-)')
ylabel('Error (%)')

% samp = 1 + mc.samples;
% dcd = samp(1,:).*(samp(3,:)-samp(4,:));
% dcl = samp(2,:).*(samp(3,:)-samp(4,:));
%
% figure
% scatter(dcd,dcl,[],errPer)
% title('Error in Cost Function')
% xlabel('CD offset (%)')
% ylabel('CL offset (%)')

histogram(err)

