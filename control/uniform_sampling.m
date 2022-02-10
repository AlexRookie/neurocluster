function [xc, yc, thetac, kappac] = uniform_sampling(fit_x, fit_y)

% UNIFORM SAMPLING
% Alessandro Antonucci @AlexRookie
% University of Trento

plot_flag = false;

step = 0.4;

if plot_flag == true
    figure(303);
    axis equal, grid on, hold on;
    plot(fit_x, fit_y);
end

SP = PolyLine();
SP.build(fit_x, fit_y);

ncuts = ceil(SP.length()/step);
dL = SP.length()/ncuts;

xc     = nan(ncuts+1,1);
yc     = nan(ncuts+1,1);
thetac = nan(ncuts+1,1);
kappac = nan(ncuts+1,1);
for i = 0:ncuts
    s = i*dL;
    [xc(i+1), yc(i+1), thetac(i+1), kappac(i+1)] = SP.evaluate(s);
end

if plot_flag == true
    plot(xc, yc, '*', 'linestyle', 'none');
    quiver(xc, yc, 1*cos(thetac), 1*sin(thetac));
end

end