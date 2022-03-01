function hg = plot_unicycle(hg, x, y, theta, color, dimension)

% Parameters
scale        = dimension;
L            = 0.35*scale; % vehicle length
W            = 0.4*scale;  % vehicle width
linewidth    = 1;          % width of the plot line
arrow_length = 0.1*scale;
wheel_length = 0.3*scale;
% color        = 'k';

% Plot central arrow
R = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % rotation matrix
arrow_position = [x; y] + R * [L; 0];

daspect([1 1 1]);
hg(end+1) = plot([arrow_position(1) x], [arrow_position(2) y], 'linewidth', linewidth, 'color', color);

angle = 90 + 45;
tmp = arrow_position + [cos(theta + angle*pi/180), -sin(theta + angle*pi/180); sin(theta + angle*pi/180), cos(theta + angle*pi/180)] * [arrow_length; 0];
hg(end+1) = plot([arrow_position(1) tmp(1)], [arrow_position(2) tmp(2)], 'linewidth', linewidth, 'color', color);

angle = -angle;
tmp = arrow_position + [cos(theta + angle*pi/180), -sin(theta + angle*pi/180); sin(theta + angle*pi/180), cos(theta + angle*pi/180)] * [arrow_length; 0];
hg(end+1) = plot([arrow_position(1) tmp(1)], [arrow_position(2) tmp(2)], 'linewidth', linewidth, 'color', color);

% Plot wheel and rear axle
angle       = 90;
R           = [cos(theta + angle*pi/180), -sin(theta + angle*pi/180); sin(theta + angle*pi/180), cos(theta + angle*pi/180)]; % rotation matrix
wheel_left  = [x; y] + R * [W/2; 0];
wheel_right = [x; y] + R * [-W/2; 0];
hg(end+1) = plot([wheel_left(1) wheel_right(1)], [wheel_left(2) wheel_right(2)], 'linewidth', linewidth, 'color', color);

wheel             = wheel_left;
R                 = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % rotation matrix
wheel_front_point = wheel + R * [wheel_length/2; 0];
wheel_rear_point  = wheel + R * [-wheel_length/2; 0];
hg(end+1) = plot([wheel_front_point(1) wheel_rear_point(1)], [wheel_front_point(2) wheel_rear_point(2)], 'linewidth', linewidth, 'color', color);

wheel             = wheel_right;
wheel_front_point = wheel + R * [wheel_length/2; 0];
wheel_rear_point  = wheel + R * [-wheel_length/2; 0];
hg(end+1) = plot([wheel_front_point(1) wheel_rear_point(1)], [wheel_front_point(2) wheel_rear_point(2)], 'linewidth', linewidth, 'color', color);

hg(end+1) = circle(x+R(1,:)*[0;0],y+R(2,:)*[0.;0],.0, [1.0000 0.5490 0.0000]);
% hg(end+1) = scatter(x+R(1,:)*[0;0],y+R(2,:)*[0.2;0],'ro','filled');

end

function h = circle(x, y, r, color)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;

    % h = plot(xunit,yunit,'Color',color);
    h = patch(xunit',yunit','r','FaceColor',color,'EdgeColor','k','FaceAlpha',0.5);
end