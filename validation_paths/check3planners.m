% Compare PRM, RRT* and A* for human-like trajectory generation
% Placido Falqueto
% Alessandro Antonucci @AlexRookie
% University of Trento

clc;
close all;
clear all;

%==========================================================================

options.save = false; % save results
options.plot = true; % show plot
options.show = false; % show statistics

%==========================================================================

dim = get(0, 'Screensize');

% Load dataset trajectories
load('povo_01_2022.mat');


% Load estimated map
mapname = 'povo';
[Walls, x_lim, y_lim, map] = load_map_obstacles(mapname);

Data.Map.Walls = Walls;
Data.Map.xLim = x_lim;
Data.Map.yLim = y_lim;
Data.AxisLim = [16, 28.1, 7.5, 23];

% Plot
figure(3);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);

% Plot CAD map
%plot_obstacles('./maps/mappaRobodogC.txt', 'k');

% Plot estimated map
figure(3);%subplot(1,3,3);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    set(gcf, 'Position', [0, 0, 860, dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    ax_lims = [18, 26, 8, 17.5];
    axis(ax_lims);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 60);
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 70);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 70);
figure(2);%subplot(1,3,2);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    set(gcf, 'Position', [0, 0, 860, dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    ax_lims = [18, 26, 8, 17.5];
    axis(ax_lims);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 60);
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 70);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 70);
figure(1);%subplot(1,3,1);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    set(gcf, 'Position', [0, 0, 860, dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    ax_lims = [18, 26, 8, 17.5];
    axis(ax_lims);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 60);
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 70);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 70);

% Plot trajectories
for i = 1:numel(Humans)
    % clean human trajectories
    j = 1;
    while j <= length(Humans{i})
        if( discretize(Humans{i}(j,1), [24.88,24.91])==1 && discretize(Humans{i}(j,2), [15.58,15.61])==1 )
            Humans{i}(j,:) = [];
        else
            j = j+1;
        end
    end
%     plot(Humans{i}(:,1), Humans{i}(:,2),'LineWidth',2);
    figure(1);%subplot(1,3,1);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',4);
    figure(2);%subplot(1,3,2);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',4);
    figure(3);%subplot(1,3,3);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',4);

    generate_clothoid(map, [smooth(Humans{i}(:,1),10),smooth(Humans{i}(:,2),10)], Walls, [x_lim; y_lim], options);
    
    for jk=1:3
        figure(jk)
        set(gcf,'Units','Inches');
        pos = get(gcf,'Position');
        set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
        set(gca, 'units', 'normalized'); %Just making sure it's normalized
        Tight = get(gca, 'TightInset');  %Gives you the bording spacing between plot box and any axis labels
                                         %[Left Bottom Right Top] spacing
        NewPos = [Tight(1) Tight(2)+0.06 1-Tight(1)-Tight(3) 1-Tight(2)-(Tight(4)+0.08)]; %New plot position [X Y W H]
        set(gca, 'Position', NewPos);
        print(gcf,strcat('figures/validation',num2str(jk)),'-dpdf','-r0');
        saveas(gcf,strcat('figures/validation',num2str(jk)));
    end


    break;%pause(2);
    clf(figure(3))
    figure(3);%subplot(1,3,3);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    figure(2);%subplot(1,3,2);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    figure(1);%subplot(1,3,1);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});

end


disp('Done');