function samples = clothoids_PRM_montecarlo_map(num_traj, num_points, pos, ob_map, options)

obstacles = ob_map.obstacles;
res = ob_map.res;
map = ob_map.map_res;

LVEC = 0.5;

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

% Find a valid area to generate starting/ending points
area = montecarlo_area(map, res, pos);

for i = 1:num_traj
    if mod(i,50) == 0
        disp(i);
    end
    
    prm = mobileRobotPRM(map, 100);
 
    % Generate starting/ending points and angles randomly
    P1 = [(area.x1(2)-area.x1(1))*rand()+area.x1(1), (area.y1(2)-area.y1(1))*rand()+area.y1(1)];
    a1 = pos.a1 + rand()*pi/8-pi/16;
    P2 = [(area.x2(2)-area.x2(1))*rand()+area.x2(1), (area.y2(2)-area.y2(1))*rand()+area.y2(1)];
    a2 = pos.a2 + rand()*pi/8-pi/16;
    
    % Plan a path between P1 and P2 using PRM
    path = findpath(prm,P1,P2);

    path(2,:)=[];
    path(end-1,:)=[];
    path = path(1:ceil(length(path)/6):end,:);

    % Plot the PRM path
    % plot(path(:,1),path(:,2),'LineWidth',2)

    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:) = P2;
    else
        path(end+1,:) = P2;
    end
    
    % Build the spline using the points planned with PRM until there is no
    % collisions
    collision = true;
    while collision

        npts = 1000;
        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:size(obstacles,1)
            for j = 2:size(obstacles{k,1},2)
                L = LineSegment(obstacles{k,1}(:,j-1), obstacles{k,1}(:,j));
                if SL.collision(L)
                    collision = true;
                    % Find the coordinates of the collision
                    [X,Y]=SL.eval(SL.intersect(L));
                    % Determine which of the spline segments collided
                    i_coll = SL.closestSegment(mean(X),mean(Y)) + 1;
                    % Add a contraint point in the colliding segment
                    path = [path(1:i_coll,:); mean(path(i_coll:i_coll+1,1)),mean(path(i_coll:i_coll+1,2)); path(i_coll+1:end,:)];
                    % error('Spline has a collision!');
                    break;
                end
            end
            if collision
                break;
            end
        end
    end
    
    if options.plot
        % SL.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});
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