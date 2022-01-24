close all;
clear all;
clc;

% Parameters
num_traj   = 10;                         % number of trajectories
num_points = 100;                        % MINIMUM number of points
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'test';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'

options.save = false;
options.plot = true;

% Folder tree
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));
addpath(genpath('./C2xyz_v2/'));

colors = customColors;

%% Generate data

% Call path clustering
samples = call_generators(generator, map, num_traj, num_points, options);

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x (m)');
xlabel('y (m)');
title('Dataset');
cellfun(@plot, samples.x, samples.y);
%for i =1:num_traj
%    plot(squeeze(samples(i,1,:)), squeeze(samples(i,2,:)));
%end

% figure(2);
% hold on, grid on, box on, axis equal;
% xlabel('x (m)');
% xlabel('y (m)');
% cellfun(@(X) plot(X(:,1), X(:,2)), Data.Humans);

%% Process data

% Get length vectors
% for i = 1:num_traj
%     [samples.length(i,:), ~, ~] = curvature([samples.x(i,:); samples.y(i,:)]');
% end
% nexttile;
% plot(1:num_points, 1./R);
% grid on;
% nexttile;
% plot(samples.x(1,:), samples.y(1,:));
% axis equal
% grid on
% hold on
% quiver(samples.x(1,:)', samples.y(1,:)', k(:,1),k(:,2));

% w = 10;
% a = [];
% b = [];
% for k = 1+w:num_points-w
%     i = 1;
%     a(end+1,:) = samples.x(i,k-w:k);
%     b(end+1,:) = samples.x(i,k:k+w);
% end

% Plot
figure(2);
tiledlayout(4,4, 'Padding', 'none', 'TileSpacing', 'compact');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.x);
title('x');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dx);
title('dx');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.ddx);
title('ddx');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dddx);
title('dddx');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.y);
title('y');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dy);
title('dy');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.ddy);
grid on;
title('ddy');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dddy);
title('dddy');
nexttile;
hold on, grid on;
cellfun(@(X,Y) plot(X, Y .* 180/pi), samples.s, samples.theta);
ylim([-180, 180]);
title('theta');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dtheta);
title('dtheta');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.ddtheta);
title('ddtheta');
nexttile;
hold on, grid on;
cellfun(@plot, samples.s, samples.dddtheta);
title('dddtheta');

%%

% Get area limits
area_file = [map, '_areas'];
limits = cell(0);
if not(isempty(area_file))
    % Open file
    fid = fopen([area_file,'.txt'],'r');
    if not(fid == -1)
        while ~feof(fid)
            line = fgets(fid);
            pts_ini = sscanf(line, '%f %f');
            line = fgets(fid);
            pts_fin = sscanf(line, '%f %f');
            % Segment vertices
            limits(end+1,1) = {[pts_ini(1), pts_fin(1); pts_ini(2), pts_fin(2)]};
        end
        % Close file
        fclose(fid);
    else
        disp(['Impossible to open file: ', area_file]);
    end
end

% Find area limits crossing
limits_on_traj = cell(num_traj,1);
for i = 1:num_traj
    jj = 1;
    for j = 1:length(samples.s{i})
        min_d = Inf;
        min_j = NaN;
        for k = 1:numel(limits)
            a = limits{k}(:,1)' - limits{k}(:,2)';
            b = [samples.x{i}(j), samples.y{i}(j)] - limits{k}(:,2)';
            a(3) = 0;
            b(3) = 0;
            d = norm(cross(a,b)) / norm(a);
            if d < min_d && d < 0.05
                min_d = d;
                min_j = j;
            end
        end
        if not(isnan(min_j))
            limits_on_traj{i}(jj) = min_j;
            jj = jj+1;
        end
    end
end
% Remove subsequent points
for i = 1:num_traj
    to_del = [];
    for j = 1:length(limits_on_traj{i})-1
        if (limits_on_traj{i}(j)+1 == limits_on_traj{i}(j+1))
            to_del = [to_del, j];
        end
    end
    limits_on_traj{i}(to_del) = [];
end


%limits_on_traj = limits_on_traj(all(~isnan(limits_on_traj)));

% Plot

figure(100);
for k = 1:numel(limits)
    plot(limits{k}(1,:), limits{k}(2,:), '--', 'linewidth', 1);
end
for i = 1:num_traj
    plot(samples.x{i}(limits_on_traj{i}), samples.y{i}(limits_on_traj{i}), '.', 'color', colors{i}, 'markersize', 26, 'linestyle', 'none');
end

figure(2);
nexttile(1);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.x, num2cell(1:num_traj));
nexttile(2);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.dx, num2cell(1:num_traj));
nexttile(3);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.ddx, num2cell(1:num_traj));
nexttile(4);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.ddx, num2cell(1:num_traj));
nexttile(5);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.y, num2cell(1:num_traj));
nexttile(6);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.dy, num2cell(1:num_traj));
nexttile(7);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.ddy, num2cell(1:num_traj));
nexttile(8);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.dddy, num2cell(1:num_traj));
nexttile(9);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.theta, num2cell(1:num_traj));
nexttile(10);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.dtheta, num2cell(1:num_traj));
nexttile(11);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.ddtheta, num2cell(1:num_traj));
nexttile(12);
cellfun(@(X,Y,I) plot(X(limits_on_traj{I}), Y(limits_on_traj{I}), '.', 'color', colors{I}, 'markersize', 24, 'linestyle', 'none'), samples.s, samples.dddtheta, num2cell(1:num_traj));



