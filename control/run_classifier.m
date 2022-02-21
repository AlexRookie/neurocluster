% run classifier

%% Create model

dt = 0.1;
T = (length(samples.s{1})-1)*dt;

xt.time = (0:dt:T)';
yt.time = (0:dt:T)';
thetat.time = (0:dt:T)';
kappat.time = (0:dt:T)';

xt.signals.values = samples.x{:}';
yt.signals.values = samples.y{:}';
thetat.signals.values = samples.theta{:}';
kappat.signals.values = samples.kappa{:}';

% Give clean data
% dt = 0.1;
% T = (size(X,1)-1)*dt;
% 
% xt.time = (0:dt:T)';
% yt.time = (0:dt:T)';
% thetat.time = (0:dt:T)';
% kappat.time = (0:dt:T)';
% 
% xt.signals.values = squeeze(X(:,1,:));
% yt.signals.values = squeeze(X(:,2,:));
% thetat.signals.values = squeeze(X(:,3,:));
% kappat.signals.values = squeeze(X(:,4,:));

fprintf('Simulation time: %f.\n',T);

%% Run simulink model

disp('Running model...');
clear out;
out = sim('class_control',T);

%% Analyse data

class_names = {'L', 'R', 'S'};

figure(100);
for i = 1:out.x.Length
    if (out.valid.Data(i))
        plot(out.x.Data(i,end), out.y.Data(i,end), '*');
        text(out.x.Data(i,end)+0.2, out.y.Data(i,end)-0.3, [class_names{out.class.Data(i)}, ' ', num2str(round(out.conf.Data(i,out.class.Data(i))*100))], 'color', 'g', 'fontsize', 12);
    end
end