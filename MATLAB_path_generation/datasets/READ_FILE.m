% Analise dataset
% Alessandro Antonucci @AlexRookie
% Placifo Falqueto
% University of Trento

close all;
clear all;
clc;

%==========================================================================

options.save = false; % save results
options.plot = false; % show plot
options.show = false; % show statistics

% Folder tree
addpath(genpath('../functions/'));
addpath(genpath('../Clothoids/'));

colors = customColors;

%==========================================================================

% Dataset file
files = dir(fullfile('*.mat'));

load('edinburgh_10Sep.mat');
f=1;
files(f).name = 'edinburgh_10Sep.mat';

totalMeanL = [];
totalMinL = [];
totalMaxL = [];
totalRMSE = [];
totalERR = [];
totalRMSE_seg = [];
totalERR_seg = [];

%for f = 1:numel(files)
    disp(['Processing ', files(f).name, ' set...']);
    
    % Load data
    load(files(f).name);
    Humans = Data.Humans;
    
    % Plot dataset
    %{
    figure(2);
    hold on, grid on, box on, axis equal;
    axis(Data.AxisLim);
    xlabel('x (m)');
    xlabel('y (m)');
    title(files(f).name, 'interpreter', 'latex');
    %cellfun(@(X) plot(X(:,1), X(:,2)), Humans);
    drawnow;
    %}
  
    LVEC = 0.5;
    
    L = [];
    RMSE = NaN(1,numel(Humans));
    ERR = NaN(1,numel(Humans));
    RMSE_seg = NaN(1,numel(Humans));
    ERR_seg = NaN(1,numel(Humans));
    
    for i = 1:numel(Humans)
        if mod(i,100)==0
            fprintf("%d/%d\n", i, numel(Humans));
        end
        
        if options.plot
            clf(figure(1));
            hold on, grid on, box on, axis equal;
            axis(Data.AxisLim);
            xlabel('x (m)');
            xlabel('y (m)');
            title(files(f).name, 'interpreter', 'latex');
        end
        
        
        HS = [smooth(Humans{i}(:,1),100), smooth(Humans{i}(:,2),100)];
        HSdx = smooth(diff(HS(:,1)),10);
        HSdy = smooth(diff(HS(:,2)),10);
        
        if options.plot
            plot(Humans{i}(:,1), Humans{i}(:,2), 'LineWidth', 2);
            plot(HS(:,1), HS(:,2), 'LineWidth', 2, 'Color', 'red');
        end
            
        %{
        figure(2)
        subplot(2,2,1)
        plot(Humans{i}(:,1))
        hold on;
        grid on;
        box on;
        plot(HS(:,1))
        subplot(2,2,2)
        plot(Humans{i}(:,2))
        hold on;
        grid on;
        box on;
        plot(HS(:,2))
        subplot(2,2,3)
        plot(HSdx)
        hold on;
        grid on;
        box on;
        subplot(2,2,4)
        plot(HSdy)
        hold on;
        grid on;
        box on;
        figure(1)
        %}
        
        smoothangle = 10;
        
        P1 = [Humans{i}(1,1), Humans{i}(1,2)];
        a1 = atan2(mean(HS(1:smoothangle,2))-P1(2), mean(HS(1:smoothangle,1))-P1(1));
        P2 = [Humans{i}(end,1), Humans{i}(end,2)];
        a2 = atan2(P2(2)-mean(HS(end-smoothangle:end,2)), P2(1)-mean(HS(end-smoothangle:end,1)));
            
        if options.plot
            plot(P1(1), P1(2), 'ro', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
            
            plot(P2(1), P2(2), 'bo', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','y',...
                'MarkerSize',5);
            
            quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'black' );
            quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'black' );
        end
        
        zerocrossing = 0.01;
        
        K = dsearchn(HS, [(max(HS(:,1))+min(HS(:,1)))/2, (max(HS(:,2))+min(HS(:,2)))/2]);
        KK1 = find(HSdx<zerocrossing & HSdx>-zerocrossing);
        KK1d = [];
        
        if length(KK1)>0
            check = KK1(1);
            for z = 2:length(KK1)
                if KK1(z)-1 == KK1(z-1)
                    check = [check KK1(z)];
                end
                if KK1(z)-1 ~= KK1(z-1) | z==length(KK1)
                    minimo = [abs(HSdx(check(1))), check(1)];
                    for j = 2:length(check)
                        if abs(HSdx(check(j))) < minimo(1)
                            minimo = [abs(HSdx(check(j))), check(j)];
                        end
                    end
                    KK1d = [KK1d; minimo(2)];
                    check = KK1(z);
                end
            end
        end
        KK1d(KK1d<6|KK1d>length(HS)-5) = [];
        
        KK2 = find(HSdy<zerocrossing & HSdy>-zerocrossing);
        KK2d = [];
        if length(KK2)>0
            check = KK2(1);
            for z = 2:length(KK2)
                if KK2(z)-1 == KK2(z-1)
                    check = [check KK2(z)];
                end
                if KK2(z)-1 ~= KK2(z-1) | z==length(KK2)
                    minimo = [abs(HSdy(check(1))), check(1)];
                    for j = 2:length(check)
                        if abs(HSdy(check(j))) < minimo(1)
                            minimo = [abs(HSdy(check(j))), check(j)];
                        end
                    end
                    KK2d = [KK2d; minimo(2)];
                    check = KK2(z);
                end
            end
        end
        KK2d(KK2d<6|KK2d>length(HS)-5) = [];
        
        KKd = unique([K; KK1d; KK2d]);
        
        MM = HS(KKd,:);
        [C,IA,IC] = unique(MM,'rows');
        MM = MM(sort(IA),:);
                
        if options.plot
            plot(MM(:,1), MM(:,2), 'ro', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
        end
        
        M12 = HS(K,:);
        
        if options.plot
            plot(M12(1), M12(2), 'ro', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
        end
        
        d1 = sqrt((MM(1,1)-P1(1))^2+(MM(1,2)-P1(2))^2);
        d2 = sqrt((MM(end,1)-P1(1))^2+(MM(end,2)-P1(2))^2);
        if(d1<d2)
            path = [P1; MM; P2];
        else
            path = [P1; flipud(MM); P2];
        end
        
        %------------------------------------------------------------------
        
        % Segment
        seg = [];
        num_of_points = [0; KKd; size(HS,1)];
        for j = size(path,1):-1:2
            nop = num_of_points(j)-num_of_points(j-1);
            seg = [seg; [linspace(path(j,1),path(j-1,1),nop); linspace(path(j,2),path(j-1,2),nop)]'];
        end
        seg = flipud(seg);
        
        if options.plot
            plot(seg(:,1), seg(:,2), '.', 'LineWidth', 2, 'Color', 'm');
        end
         
        dist = [];
        for s = 1:length(HS)
            d = norm(HS(s,:) - seg(s,:));
            dist = [dist; d];
        end
        
        RMSE_seg(i) = sqrt(mean(dist.^2));
        ERR_seg(i) = max(dist);
        
        %------------------------------------------------------------------
        
        to_del = [];
        for j = 2:size(path,1)-1
            if norm(path(j-1,:)-path(j,:)) < 0.1
                to_del = [to_del, j];
            end
        end
        path(to_del,:) = [];
        
        % Clothoid
        npts = 100;
        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        
        if options.plot
            SL.plot(npts, {'Color','#77AC30','LineWidth',2}, {'Color','#77AC30','LineWidth',2});
        end
                
        l = SL.length();
        n = size(HS,1);
        SL_sampled = NaN(n,2);
        for k = 1:n
            sl = (k-1)*l/(n-1);
            [SL_sampled(k,1), SL_sampled(k,2), ~] = SL.evaluate(sl);
        end
        
        dist = [];
        for s = 1:length(HS)
            d = norm(HS(s,:) - SL_sampled(s,:));
            %d = SL.distance(HS(s,1),HS(s,2));
            dist = [dist; d];
            
           % plot(HS(s,1), HS(s,2), '*', 'markersize', 20, 'linestyle', 'none')
           % plot(SL_sampled(s,1), SL_sampled(s,2), '*', 'markersize', 12, 'linestyle', 'none')
        end
        
        RMSE(i) = sqrt(mean(dist.^2));
        ERR(i) = max(dist);
        
        for k = 1:SL.numSegments()
            L = [L, SL.get(k).length()];
        end
        
        if options.show
            [RMSE(i), RMSE_seg(i), ERR(i), ERR_seg(i)]
        end
        
    end
    
    pause();
    
    totalRMSE = [totalRMSE, nanmean(RMSE)];
    totalERR = [totalERR, nanmean(ERR)];
    totalMeanL = [totalMeanL, nanmean(L)];
    totalMinL = [totalMinL, nanmin(L)];
    totalMaxL = [totalMaxL, nanmax(L)];
    
    totalRMSE_seg = [totalRMSE_seg, nanmean(RMSE_seg)];
    totalERR_seg = [totalERR_seg, nanmean(ERR_seg)];

%end

if options.show
    [totalRMSE, totalRMSE_seg, totalERR, totalERR_seg]    
   options figure(3);
    hold on;
    plot(RMSE);
    plot(RMSE_seg, '--');
    plot(ERR);
    plot(ERR_seg, '--');
end

disp('Done!');
