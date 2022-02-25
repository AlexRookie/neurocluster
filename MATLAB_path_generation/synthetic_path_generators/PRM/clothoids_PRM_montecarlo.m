function clothoids = clothoids_PRM_montecarlo(Map, Pos, num_traj, randomize)

area = Map.area;
poly_obstacles = Map.poly_obstacles;
map_res = Map.map_res;

l_vec = 0.5; % orientation angle length (for plot)
options_plot = true; % plot flag

time = clock();
rng(time(6));

clothoids = cell(1,num_traj);

i = 1;
loopa = true;
while loopa
    if mod(i,50) == 0
        disp(i);
    end
       
    valid = true;
    prm = mobileRobotPRM(map_res, 200);

    % Generate starting/ending points and angles (randomly) inside the valid area
    if randomize == true
        in=0;
        while ~in
            P1 = [unifrnd( min(area.c1.Vertices(:,1)), max(area.c1.Vertices(:,1)) ), unifrnd( min(area.c1.Vertices(:,2)), max(area.c1.Vertices(:,2)) )];
            in = isinterior(area.c1,P1);
        end
        in=0;
        while ~in
            P2 = [unifrnd( min(area.c2.Vertices(:,1)), max(area.c2.Vertices(:,1)) ), unifrnd( min(area.c2.Vertices(:,2)), max(area.c2.Vertices(:,2)) )];
            in = isinterior(area.c2,P2);
        end
        a1 = Pos.a1 + rand()*pi/8-pi/16;
        a2 = Pos.a2 + rand()*pi/8-pi/16;
    else
        P1 = [Pos.x1, Pos.y1];
        P2 = [Pos.x2, Pos.y2];
        a1 = Pos.a1;
        a2 = Pos.a2;
    end
    
    % Plan a path between P1 and P2 using PRM
    path = findpath(prm,P1,P2);

    path(2,:)=[];
    path(end-1,:)=[];
    path = path(1:ceil(length(path)/6):end,:);

    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:) = P2;
    else
        path(end+1,:) = P2;
    end
       
    to_del = []; % remove consecutive points if d <= 1e-3
    for k = 2:max(size(path))
        if (norm([path(k,:)-path(k-1,:)]) <= 1e-3)
            to_del = [to_del, k];
        end
    end
    path(to_del,:) = [];
    
    % Plot the PRM path
    % plot(path(:,1),path(:,2),'LineWidth',2)
    
    % Build the spline using the points planned with PRM until there is no collisions
    counter = 1;
    collision = true;
    while collision

        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:max(size(poly_obstacles))
            for j = 2: max(size(poly_obstacles{k}.Vertices))
                L = LineSegment(poly_obstacles{k}.Vertices(j-1,:), poly_obstacles{k}.Vertices(j,:));
                if SL.collision(L)
                    collision = true;
                    
                    % Find the coordinates of the collision
                    [X,Y]=SL.eval(SL.intersect(L));
                    % Determine which of the spline segments collided
                    i_coll = SL.closestSegment(mean(X),mean(Y)) + 1;
                    % Add a constraint point in the colliding segment
                    path = [path(1:i_coll,:); mean(path(i_coll:i_coll+1,1)),mean(path(i_coll:i_coll+1,2)); path(i_coll+1:end,:)];
                    % error('Spline has a collision!');
                    
                    to_del = []; % remove consecutive points if d <= 1e-3
                    for k = 2:max(size(path))
                        if (norm([path(k,:)-path(k-1,:)]) <= 1e-3)
                            to_del = [to_del, k];
                        end
                    end
                    path(to_del,:) = [];
                    break;
                end
            end
            if collision
                break;
            end
        end
        counter = counter+1;
        if counter > 20
            valid = false;
            break;
        end
    end
    
    if (valid == false)
        continue;
    end
    
    clothoids{i} = SL;
    i = i+1;
    
    if options_plot
        npts = 1000;
        SL.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});
        
        plot(P1(1), P1(2), 'ro', 'MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize',5);
        plot(P2(1), P2(2), 'bo', 'MarkerEdgeColor','k', 'MarkerFaceColor','y', 'MarkerSize',5);
        quiver( P1(1), P1(2), l_vec*cos(a1), l_vec*sin(a1), 'Color', 'r' );
        quiver( P2(1), P2(2), l_vec*cos(a2), l_vec*sin(a2), 'Color', 'r' );
    end
    
    if i > 1:num_traj
        loopa = false;
    end
end

end