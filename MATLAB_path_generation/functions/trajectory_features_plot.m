function fig = trajectory_features_plot(traj)

% Plot
fig = figure(2);
tiledlayout(4,4, 'Padding', 'none', 'TileSpacing', 'compact');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.x);
title('x');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dx);
title('dx');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.ddx);
title('ddx');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dddx);
title('dddx');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.y);
title('y');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dy);
title('dy');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.ddy);
grid on;
title('ddy');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dddy);
title('dddy');
nexttile;
hold on, grid on;
cellfun(@(X,Y) plot(X, Y .* 180/pi), traj.s, traj.theta);
ylim([-180, 180]);
title('theta');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dtheta);
title('dtheta');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.ddtheta);
title('ddtheta');
nexttile;
hold on, grid on;
cellfun(@plot, traj.s, traj.dddtheta);
title('dddtheta');

end