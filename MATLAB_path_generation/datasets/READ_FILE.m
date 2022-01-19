% Parse txt dataset and create .mat files
% Alessandro Antonucci @AlexRookie
% University of Trento

clc;
clear all;
%close all;

%==========================================================================

% Dataset parameters
convers = 0.0247; % 1 pixel = 0.0247 m
dt = 1/9; % average sampling time

saveflag = true; % save results
plotflag = true; % show plot

%==========================================================================

load('thor3.mat');
Humans = Data.Humans;
    
    % Plot
    if plotflag == true

        LVEC = 0.5;

        figure(1);
        
        RMSE = [];

        for i = 1:numel(Humans)
            if i == 14
                break;
            end
%             clf(figure(2));
            clf(figure(1));
            hold on;
            grid on;
            box on;
            axis equal;
            axis equal;
            axis(Data.AxisLim)

            HS = [smooth(Humans{i}(:,1),100), smooth(Humans{i}(:,2),100)];
            HSdx = smooth(diff(HS(:,1)),10);
            HSdy = smooth(diff(HS(:,2)),10);

            plot(Humans{i}(:,1), Humans{i}(:,2),'LineWidth',2);
            plot(HS(:,1), HS(:,2), 'LineWidth',2,'Color','red');

%             figure(2)
%             subplot(2,2,1)
%             plot(Humans{i}(:,1))
%             hold on;
%             grid on;
%             box on;
%             plot(HS(:,1))
%             subplot(2,2,2)
%             plot(Humans{i}(:,2))
%             hold on;
%             grid on;
%             box on;
%             plot(HS(:,2))
%             subplot(2,2,3)
%             plot(HSdx)
%             hold on;
%             grid on;
%             box on;
%             subplot(2,2,4)
%             plot(HSdy)
%             hold on;
%             grid on;
%             box on;
% 
%             figure(1)

            smoothangle = 20;

            P1 = [Humans{i}(1,1), Humans{i}(1,2)];
            a1 = atan2(mean(HS(1:smoothangle,2))-P1(2), mean(HS(1:smoothangle,1))-P1(1));
            P2 = [Humans{i}(end,1), Humans{i}(end,2)];
            a2 = atan2(P2(2)-mean(HS(end-smoothangle:end,2)), P2(1)-mean(HS(end-smoothangle:end,1)));
            

        
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
            plot(MM(:,1), MM(:,2), 'ro', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);

            M12 = HS(K,:);

            plot(M12(1), M12(2), 'ro', ...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
    
            d1 = sqrt((MM(1,1)-P1(1))^2+(MM(1,2)-P1(2))^2);
            d2 = sqrt((MM(end,1)-P1(1))^2+(MM(end,2)-P1(2))^2);
            if(d1<d2)
                path = [P1; MM; P2];
            else
                path = [P1; flipud(MM); P2];
            end
            
            npts = 100;
            CL = ClothoidSplineG2();
            CL.verbose(false);
            SL = CL.buildP1(path(:,1), path(:,2), a1, a2);

            SL.plot(npts,{'Color','#77AC30','LineWidth',2},{'Color','#77AC30','LineWidth',2});
            pause(0.5);

            dist = [];
            for s = 1:length(HS)
                d = SL.distance(HS(s,1),HS(s,2));
                dist = [dist; d];
            end

            RMSE = [RMSE; sqrt(mean(dist.^2))];
            
        end

        totalRMSE = mean(RMSE)

    end
    % 

%    pause();
    
%end

disp('Done!');
