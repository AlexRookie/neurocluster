% Test clothoids

clc;
close all;
clear all;

% Folder tree
addpath(genpath('./Clothoids/'));

x1 = 0;
y1 = 0;
angle1 = 0;

x2 = 2;
y2 = 0;
angle2 = 0;

x3 = 5;
y3 = 0;
angle3 = 0;

%C = ClothoidCurve();
%iter = C.build_G1(x1, y1, angle1, x2, y2, angle2);

C = ClothoidList();
C.push_back_G1(x1, y1, angle1, x2, y2, angle2);
C.push_back_G1(x2, y2, angle2, x3, y3, angle3);

figure(1);
subplot(1,2,1);
C.plot(1000, {'LineWidth', 2, 'Color', 'm'}, {'LineWidth', 2, 'Color', 'g'});
axis equal, grid on, box on;
[xmin, ymin, xmax, ymax] = C.bbox();
xlim([xmin-1, xmax+1]);
ylim([ymin-1, ymax+1]);

L = C.length();
[x, y, theta, kappa] = C.evaluate(linspace(0,L,1000));
%[x, y] = C.points(1000);
[dx, dy] = C.eval_D(linspace(0,L,1000));
dtheta = C.theta_D(linspace(0,L,1000));
dkappa = C.kappa_D(linspace(0,L,1000));

%[X,Y] = C.points(1000);

figure(1);
subplot(1,2,2);
plot(x, y, '-', 'LineWidth', 2, 'Color', 'r');
axis equal, grid on, box on;
xlim([xmin-1, xmax+1]);
ylim([ymin-1, ymax+1]);

figure(2);
subplot(2,3,1);
hold on, grid on, box on;
plot(x, 'LineWidth', 2);
plot(y, 'LineWidth', 2);
legend({'x','y'});
subplot(2,3,2);
hold on, grid on, box on;
plot(theta*180/pi, 'LineWidth', 2);
plot(kappa, 'LineWidth', 2);
legend({'theta','kappa'});
subplot(2,3,4);
hold on, grid on, box on;
plot(dx, 'LineWidth', 2);
plot(dy, 'LineWidth', 2);
legend({'dx/ds','dy/ds'});
subplot(2,3,5);
hold on, grid on, box on;
plot(dtheta, 'LineWidth', 2);
plot(dkappa, 'LineWidth', 2);
legend({'dtheta','dkappa'});

