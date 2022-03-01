function generate_clothoid(map,traj,obstacles,lims,options)

    LVEC = 0.5;
    smoothangle = round(0.2*length(traj));
    skip_samples = round(smoothangle/2);

    map1 = copy(map);
    inflate(map1,0.5);
    
    P1 = [traj(1,1), traj(1,2)];
    a1 = atan2(mean(traj(skip_samples:skip_samples+smoothangle,2))-P1(2), mean(traj(skip_samples:skip_samples+smoothangle,1))-P1(1));
    P2 = [traj(end,1), traj(end,2)];
    a2 = atan2(P2(2)-mean(traj(end-smoothangle-skip_samples:end-skip_samples,2)), P2(1)-mean(traj(end-smoothangle-skip_samples:end-skip_samples,1)));

    if options.plot
        figure(1);%subplot(1,3,1);
        plot(P1(1), P1(2), 'ro', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','g',...
            'MarkerSize',5);
        
        plot(P2(1), P2(2), 'bo', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','y',...
            'MarkerSize',5);
        
        quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'black' );
        quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'black' );

        figure(2);%subplot(1,3,2);
        plot(P1(1), P1(2), 'ro', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','g',...
            'MarkerSize',5);
        
        plot(P2(1), P2(2), 'bo', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','y',...
            'MarkerSize',5);
        
        quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'black' );
        quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'black' );

        figure(3);%subplot(1,3,3);
        plot(P1(1), P1(2), 'ro', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','g',...
            'MarkerSize',5);
        
        plot(P2(1), P2(2), 'bo', ...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','y',...
            'MarkerSize',5);
        
        quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'black' );
        quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'black' );
    end

    % PRM planner =====================================================================

    prm = mobileRobotPRM(map1, 500); %500

    path = findpath(prm,[P1(1)-lims(1), P1(2)-lims(2)],[P2(1)-lims(1), P2(2)-lims(2)]);

    path(2,:)=[];
    path(end-1,:)=[];
    path = path(1:ceil(length(path)/6):end,:);

    path = [path(:,1)+lims(1),path(:,2)+lims(2)];

    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:) = P2;
    else
        path(end+1,:) = P2;
    end

    figure(1);%subplot(1,3,1);
    % Plot the PRM path
    plot(path(:,1),path(:,2),'LineWidth',4)

    % Build the spline using the points planned with PRM until there is no collisions
    collision = true;
    while collision

        npts = 1000;
        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:size(obstacles,1)
            for j = 2:size(obstacles{k,2},2)
                L = LineSegment(obstacles{k,2}(:,j-1), obstacles{k,2}(:,j));
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

    genTraj = zeros(length(traj),2);
    for i = 1:length(traj)
        [genTraj(i,1), genTraj(i,2)] = SL.eval((i-1)*SL.length/(length(traj)-1));
    end

    distancePRM = DiscreteFrechetDist(genTraj,traj)
    % text(18,10,num2str(distance),'horizontal','center','color','k','FontSize',14);

    % Compare Frechet Distance to maximum of minimum distances
%     dist = [];
%     for i = 1:length(traj)
%         d = SL.distance(traj(i,1),traj(i,2));
%         dist = [dist; d];
%     end
%     
%     max_dist = max(dist)
    
    if options.plot
        SL.plot(npts,{'Color','blue','LineWidth',4},{'Color','blue','LineWidth',4});
    end

    % RRT* planner =====================================================================

    ss = stateSpaceSE2;
    ss.StateBounds = [map1.XWorldLimits; map1.YWorldLimits; [-pi pi]];
    
    sv = validatorOccupancyMap(ss); 
    sv.Map = map1;
    sv.ValidationDistance = 0.01;
    
    planner = plannerRRTStar(ss,sv);
    planner.MaxConnectionDistance = 3.0;
    planner.MaxIterations = 1e4;
    planner.MaxNumTreeNodes = 1e4;
    planner.ContinueAfterGoalReached = true;

    [pthObj, solnInfo] = plan(planner,[[P1(1)-lims(1), P1(2)-lims(2)] a1],[[P2(1)-lims(1), P2(2)-lims(2)] a2]);
        
    path = pthObj.States(1:round(length(pthObj.States)/6):end,:);
    path = [path(:,1)+lims(1),path(:,2)+lims(2),path(:,3)];
    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:)=[P2 a2];
    else
        path(end+1,:)=[P2 a2];
    end

    figure(2);%subplot(1,3,2);
    % Plot the RRT* path
    plot(path(:,1),path(:,2),'LineWidth',4)

    % Build the spline using the points planned with PRM until there is no collisions
    collision = true;
    while collision

        npts = 1000;
        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:size(obstacles,1)
            for j = 2:size(obstacles{k,2},2)
                L = LineSegment(obstacles{k,2}(:,j-1), obstacles{k,2}(:,j));
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

    genTraj = zeros(length(traj),2);
    for i = 1:length(traj)
        [genTraj(i,1), genTraj(i,2)] = SL.eval((i-1)*SL.length/(length(traj)-1));
    end

    distanceRRT = DiscreteFrechetDist(genTraj,traj)
    % text(18,10,num2str(distance),'horizontal','center','color','k','FontSize',14);

    % Compare Frechet Distance to maximum of minimum distances
%     dist = [];
%     for i = 1:length(traj)
%         d = SL.distance(traj(i,1),traj(i,2));
%         dist = [dist; d];
%     end
%     
%     max_dist = max(dist)
    
    if options.plot
        SL.plot(npts,{'Color','blue','LineWidth',4},{'Color','blue','LineWidth',4});
    end

    % A* planner =====================================================================

    
    planner = plannerAStarGrid(occupancyMap(flip(map1.occupancyMatrix)));

%     path = plan(planner,floor([27-P1(2)*res P1(1)*res]),floor([27-P2(2)*res P2(1)*res]));
%     path = [path(:,2) 27-path(:,1)]./res;

%     figure
%     planner.show
    round([P1(1)-lims(1), P1(2)-lims(2)].*map1.Resolution)
    round([P2(1)-lims(1), P2(2)-lims(2)].*map1.Resolution)


    path = plan(planner,ceil([P1(1)-lims(1), P1(2)-lims(2)].*map1.Resolution),...
                        ceil([P2(1)-lims(1), P2(2)-lims(2)].*map1.Resolution));

    path = path./map1.Resolution;    
    path = [path(:,1)+lims(1),path(:,2)+lims(2)];
    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:)=P2;
    else
        path(end+1,:)=P2;
    end

    figure(3);%subplot(1,3,3);
    % Plot the A* path
    plot(path(:,1),path(:,2),'LineWidth',4)

    detectChanges = diff(diff(path));
    detectChanges = abs(detectChanges);
    detectChanges = detectChanges > 0.01;
    detectChanges = detectChanges(:,1) | detectChanges(:,2);
    detectChanges(1) = 1;
    detectChanges(end) = 1;

    middle = 0;
    flagVariation = 0;
    VariationStart = 0;
    VariationEnd = 0;
    for ik = 1:length(detectChanges)
        if detectChanges(ik) == 1
            if ~flagVariation
                flagVariation = 1;
                VariationStart = ik;
                VariationEnd = ik;
            else
                VariationEnd = ik;
            end
        elseif flagVariation
            flagVariation = 0;
            middle = ceil((VariationEnd+VariationStart)/2);
            detectChanges([VariationStart:middle-1,middle+1:VariationEnd]) = 0;
        end
    end
    path(~detectChanges,:) = [];


    % Build the spline using the points planned with PRM until there is no collisions
    collision = true;
    while collision

        npts = 1000;
        CL = ClothoidSplineG2();
        CL.verbose(false);
        SL = CL.buildP1(path(:,1), path(:,2), a1, a2);
        collision = false;

        % Check spline-obstacles interections
        for k = 1:size(obstacles,1)
            for j = 2:size(obstacles{k,2},2)
                L = LineSegment(obstacles{k,2}(:,j-1), obstacles{k,2}(:,j));
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

    genTraj = zeros(length(traj),2);
    for i = 1:length(traj)
        [genTraj(i,1), genTraj(i,2)] = SL.eval((i-1)*SL.length/(length(traj)-1));
    end

    distanceAStar = DiscreteFrechetDist(genTraj,traj)
    % text(18,10,num2str(distance),'horizontal','center','color','k','FontSize',14);

    % Compare Frechet Distance to maximum of minimum distances
%     dist = [];
%     for i = 1:length(traj)
%         d = SL.distance(traj(i,1),traj(i,2));
%         dist = [dist; d];
%     end
%     
%     max_dist = max(dist)
    
    if options.plot
        SL.plot(npts,{'Color','blue','LineWidth',4},{'Color','blue','LineWidth',4});
    end

end