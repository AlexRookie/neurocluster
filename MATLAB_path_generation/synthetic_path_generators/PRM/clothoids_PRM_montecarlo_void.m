function samples = clothoids_PRM_montecarlo_void(num_traj, num_points, pos, map, res, options)

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

    P1 = [pos.x1 + rand()*2, pos.y1 + rand()*2]./res;
    a1 = pi/4 + rand()/8;
    P2 = [pos.x2 + rand()*2, pos.y2 + rand()*2]./res;
    a2 = pi/4 + rand()/8;
    
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
    %plot(path(:,1),path(:,2),'LineWidth',2)
    %plot(path(:,1),path(:,2),'ro');
    
    angle = a1;

    for j=1:length(path)-1

        angle0 = angle;

        C = ClothoidCurve();

        x1=path(j,1);
        y1=path(j,2);
        x2=path(j+1,1);
        y2=path(j+1,2);

        angle1 = atan2d((y2 - y1),(x2 - x1))/180*pi;

        if j+2 <= length(path)
            x3=path(j+2,1);
            y3=path(j+2,2);
            angle2 = atan2d((y3 - y2),(x3 - x2))/180*pi;
            angle = (angle1+angle2)/2;
        else
            angle = a2;
        end

        iter = C.build_G1(x1, y1, angle0, x2, y2, angle );
        if options.plot
            C.plot( 1000, '-', 'LineWidth', 2 );
        end
        
        % Get data
        L = C.length();
        [x, y, theta, kappa] = C.evaluate(linspace(0,L,num_points));
        %[a, b] = C.points(num_points); 
        [dx, dy] = C.eval_D(linspace(0,L,num_points));
        dtheta = C.theta_D(linspace(0,L,num_points));
        dkappa = C.kappa_D(linspace(0,L,num_points));
             
        %sample = [a; b];
        %samples = cat(3, samples, sample);
        
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

%samples = permute(samples,[3,1,2]);

if options.save
    fclose(fid);
end

end