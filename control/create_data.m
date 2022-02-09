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