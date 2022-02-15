function samples = get_samples(varargin) % get_samples(clothoids, step, augmentation)

options_plot = true; % plot flag

clothoids = varargin{1};
step = varargin{2};
augmentation = varargin{3};
% if (nargin >= 3)
%     num_traj = varargin{3};
% else
%     num_traj = size(clothoids,1);
% end

num_traj = max(size(clothoids));
num_rot = 1;

if augmentation == true
    num_traj = num_traj*4;
    num_rot = 4;
    cx = 10;
    cy = 10;
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

k = 1;
for i = 1:max(size(clothoids))
    % Get data
    L = clothoids{i}.length();

    for j = 1:num_rot
        samples.s{k} = 0:step:L;
        
        if j > 1
            clothoids{i}.rotate(pi/2, cx, cy);
        end
        
        [samples.x{k}, samples.y{k}, samples.theta{k}, samples.kappa{k}] = clothoids{i}.evaluate(samples.s{k});
        [samples.dx{k}, samples.dy{k}] = clothoids{i}.eval_D(samples.s{i});
        [samples.ddx{k}, samples.ddy{k}] = clothoids{i}.eval_DD(samples.s{k});
        [samples.dddx{k}, samples.dddy{k}] = clothoids{i}.eval_DDD(samples.s{k});
        samples.dtheta{k} = clothoids{i}.theta_D(samples.s{k});
        samples.ddtheta{k} = clothoids{i}.theta_DD(samples.s{k});
        samples.dddtheta{k} = clothoids{i}.theta_DDD(samples.s{k});
        samples.dkappa{k} = clothoids{i}.kappa_D(samples.s{k});
        samples.ddkappa{k} = clothoids{i}.kappa_DD(samples.s{k});
        
        k = k+1;
        
        if options_plot
            npts = 1000;
            clothoids{i}.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});
            drawnow;
        end
        
    end
    
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