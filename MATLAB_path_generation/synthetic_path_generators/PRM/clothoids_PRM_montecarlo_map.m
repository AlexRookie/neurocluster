function samples = clothoids_PRM_montecarlo_map(num_traj, num_points, map, res, options)

LVEC = 0.5/res;

time = clock();
rng(time(6));

samples.x      = NaN(num_traj, num_points);
samples.y      = NaN(num_traj, num_points);
samples.theta  = NaN(num_traj, num_points);
samples.kappa  = NaN(num_traj, num_points);
samples.dx     = NaN(num_traj, num_points);
samples.dy     = NaN(num_traj, num_points);
samples.dtheta = NaN(num_traj, num_points);
samples.dkappa = NaN(num_traj, num_points);

% Save data
if options.save
    FileName = "Prova.txt";
    fid = fopen(FileName, 'w');
end

for i = 1:num_traj
    if mod(i,50) == 0
        disp(i);
    end
    
    prm = mobileRobotPRM(map, 100);

    P1 = [3 + rand()*2, 1 + rand()*9]./res;
    a1 = rand()*pi/2;
    P2 = [13 + rand()*5, 3 + rand()*7]./res;
    a2 = -rand()*pi/2;

    path = findpath(prm,P1,P2);
        
    path(2,:)=[];
    path(end-1,:)=[];
    path = path(1:ceil(length(path)/6):end,:);
    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:)=P2;
    else
        path(end+1,:)=P2;
    end
    
    npts = 1000;
    CL = ClothoidSplineG2();
    SL = CL.buildP1(path(:,1), path(:,2),a1,a2);
    
    if options.plot
        SL.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});
    end

    % Get data
    L = SL.length();
    [x, y, theta, kappa] = SL.evaluate(linspace(0,L,num_points));
    [dx, dy] = SL.eval_D(linspace(0,L,num_points));
    dtheta = SL.theta_D(linspace(0,L,num_points));
    dkappa = SL.kappa_D(linspace(0,L,num_points));
    
    samples.x     (i, :) = x     ;
    samples.y     (i, :) = y     ;
    samples.theta (i, :) = theta ;
    samples.kappa (i, :) = kappa ;
    samples.dx    (i, :) = dx    ;
    samples.dy    (i, :) = dy    ;
    samples.dtheta(i, :) = dtheta;
    samples.dkappa(i, :) = dkappa;
    
    if options.save
        %fmt = [repmat('%10.5f ', 1, length(samples)-1), '%10.5f\n'];
        %fprintf(fid, fmt, samples);
    end
    
    if options.plot
        plot(P1(1), P1(2), 'ro', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','g',...
            'MarkerSize',5);
        
        plot(P2(1), P2(2), 'bo', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','y',...
            'MarkerSize',5);
        
        quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'r' );
        quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'r' );
    end
end

if options.save
    fclose(fid);
end

end