function samples = get_samples(varargin) % get_samples(clothoids, step, [num_traj])

clothoids = varargin{1};
step = varargin{2};
if (nargin >= 3)
    num_traj = varargin{3};
else
    num_traj = size(clothoids,1);
end

samples.s        = cell(1,num_traj);
samples.x        = cell(1,num_traj);
samples.y        = cell(1,num_traj);
samples.dx       = cell(1,num_traj);
samples.dy       = cell(1,num_traj);
samples.ddx      = cell(1,num_traj);
samples.ddy      = cell(1,num_traj);
samples.dddx     = cell(1,num_traj);
samples.dddy     = cell(1,num_traj);
samples.theta    = cell(1,num_traj);
samples.dtheta   = cell(1,num_traj);
samples.ddtheta  = cell(1,num_traj);
samples.dddtheta = cell(1,num_traj);
samples.kappa    = cell(1,num_traj);
samples.dkappa   = cell(1,num_traj);
samples.ddkappa  = cell(1,num_traj);

for i = 1:num_traj
  % Get data
    L = clothoids{i}.length();
    samples.s{i} = 0:step:L;
    
    %if length(samples.s{i}) < num_points
    %    error('Too few points for trajectory number %d.', i);
    %end
    
    [samples.x{i}, samples.y{i}, samples.theta{i}, samples.kappa{i}] = clothoids{i}.evaluate(samples.s{i});
    [samples.dx{i}, samples.dy{i}] = clothoids{i}.eval_D(samples.s{i});
    [samples.ddx{i}, samples.ddy{i}] = clothoids{i}.eval_DD(samples.s{i});
    [samples.dddx{i}, samples.dddy{i}] = clothoids{i}.eval_DDD(samples.s{i});
    samples.dtheta{i} = clothoids{i}.theta_D(samples.s{i});
    samples.ddtheta{i} = clothoids{i}.theta_DD(samples.s{i});
    samples.dddtheta{i} = clothoids{i}.theta_DDD(samples.s{i});
    samples.dkappa{i} = clothoids{i}.kappa_D(samples.s{i});
    samples.ddkappa{i} = clothoids{i}.kappa_DD(samples.s{i});
    
%     [x, y, theta, kappa] = SL.evaluate(linspace(0,L,num_points));
%     [dx, dy] = SL.eval_D(linspace(0,L,num_points));
%     [ddx, ddy] = SL.eval_DD(linspace(0,L,num_points));
%     [dddx, dddy] = SL.eval_DDD(linspace(0,L,num_points));
%     dtheta = SL.theta_D(linspace(0,L,num_points));
%     ddtheta = SL.theta_DD(linspace(0,L,num_points));
%     dddtheta = SL.theta_DDD(linspace(0,L,num_points));
%     dkappa = SL.kappa_D(linspace(0,L,num_points));
%     ddkappa = SL.kappa_DD(linspace(0,L,num_points));
    
%     samples.x       (i,:) = x       ;
%     samples.y       (i,:) = y       ;
%     samples.dx      (i,:) = dx      ;
%     samples.dy      (i,:) = dy      ;
%     samples.ddx     (i,:) = ddx     ;
%     samples.ddy     (i,:) = ddy     ;
%     samples.dddx    (i,:) = dddx    ;
%     samples.dddy    (i,:) = dddy    ;
%     samples.theta   (i,:) = theta   ;
%     samples.dtheta  (i,:) = dtheta  ;
%     samples.ddtheta (i,:) = ddtheta ;
%     samples.dddtheta(i,:) = dddtheta;
%     samples.kappa   (i,:) = kappa   ;
%     samples.dkappa  (i,:) = dkappa  ;
%     samples.ddkappa (i,:) = ddkappa ;

end

end