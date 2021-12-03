load exampleMaps.mat

res = 1;
map = binaryOccupancyMap(simpleMap,res);

% figure
show(map)
hold on

map1 = copy(map);
inflate(map1,0.2/res);

LVEC = 0.5/res;

% bounds = [map1.XWorldLimits; map1.YWorldLimits; [-pi pi]];
% ss = stateSpaceDubins(bounds);
ss = stateSpaceSE2;
% ss.MinTurningRadius = 3.0;
ss.StateBounds = [map1.XWorldLimits; map1.YWorldLimits; [-pi pi]];

sv = validatorOccupancyMap(ss); 
sv.Map = map1;
sv.ValidationDistance = 0.01;

planner = plannerRRTStar(ss,sv);
planner.MaxConnectionDistance = 2.0;
planner.ContinueAfterGoalReached = true;
planner.MaxIterations = 5000;
planner.MaxNumTreeNodes = 5000;

time = clock();
rng(time(6));

for i=1:10
    P1 = [3 + rand()*2, 1 + rand()*9]./res;
    a1 = rand()*pi/2;
    P2 = [13 + rand()*5, 3 + rand()*7]./res;
    a2 = -rand()*pi/2;

    plot(P1(1), P1(2), 'ro', ...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','g',...
        'MarkerSize',5);

    plot(P2(1), P2(2), 'bo', ...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','y',...
        'MarkerSize',5);


    [pthObj, solnInfo] = plan(planner,[P1 a1],[P2 a2]);

% %     Plot entire search tree.
%     plot(solnInfo.TreeData(:,1),solnInfo.TreeData(:,2),'.-');
%     
% %     Interpolate and plot path.
%     interpolate(pthObj,300)
%     plot(pthObj.States(:,1),pthObj.States(:,2),'r-','LineWidth',2)
        
    path = pthObj.States(1:floor(length(pthObj.States)/6):end,:);
    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:)=[P2 a2];
    else
        path(end+1,:)=[P2 a2];
    end
%     plot(path(:,1),path(:,2),'LineWidth',2)

%     plot(path(:,1),path(:,2),'ro');


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
        C.plot( 1000, '-b', 'LineWidth', 2 );
    end
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