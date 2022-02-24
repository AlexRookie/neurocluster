function clothoids = clothoids_Simple_montecarlo(Map, Pos, num_traj, randomize)

poly_obstacles = Map.poly_obstacles;
%res = obj_map.res;
map_res = Map.map_res;

%step = 0.05; % sampling step (cm)

l_vec = 0.5; % orientation angle length (for plot)

options_plot = true; % plot flag

time = clock();
rng(time(6));

% Find a valid area to generate starting/ending points
area = montecarlo_area(poly_obstacles, Pos);

clothoids = cell(1,num_traj);

for i = 1:num_traj
    if mod(i,50) == 0
        disp(i);
    end
    
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
    
    % Build the spline using the points planned with PRM until there is no collisions
    collision = true;
    while collision

        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1([P1(1); P2(1)], [P1(2); P2(2)], a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:size(poly_obstacles,1)
            for j = 2:size(poly_obstacles{k,1},2)
                L = LineSegment(poly_obstacles{k,1}(:,j-1), poly_obstacles{k,1}(:,j));
                if SL.collision(L)
                    collision = true;
                    
                    % Try new points
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
                end
            end
            if collision
                break;
            end
        end
    end
    
    if options_plot
        npts = 1000;
        SL.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});
    end
   
    clothoids{i} = SL;
    
    if options_plot
        plot(P1(1), P1(2), 'ro', 'MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize',5);
        plot(P2(1), P2(2), 'bo', 'MarkerEdgeColor','k', 'MarkerFaceColor','y', 'MarkerSize',5);
        quiver( P1(1), P1(2), l_vec*cos(a1), l_vec*sin(a1), 'Color', 'r' );
        quiver( P2(1), P2(2), l_vec*cos(a2), l_vec*sin(a2), 'Color', 'r' );
    end
end

end