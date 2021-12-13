load exampleMaps.mat

res = 1;
map = binaryOccupancyMap(simpleMap,res);

% figure
show(map)
hold on

map1 = copy(map);
inflate(map1,0.2/res);

LVEC = 0.8/res;

time = clock();
rng(time(6));

FileName = "Prova.txt";
fid = fopen(FileName, 'w');

for i=1:10
    prm = mobileRobotPRM(map1,100);

    P1 = [3 + rand()*2, 1 + rand()*9]./res;
    a1 = rand()*pi/2;
    P2 = [13 + rand()*5, 3 + rand()*7]./res;
    a2 = -rand()*pi/2;

    samples = [];
    angles = [];

    path = findpath(prm,P1,P2);
        
%     plot(path(:,1),path(:,2),'LineWidth',2)

    path(2,:)=[];
    path(end-1,:)=[];
    path = path(1:ceil(length(path)/6):end,:);
    d = pdist([path(end,1:2);P2],'euclidean');
    if d < 1
        path(end,:)=P2;
    else
        path(end+1,:)=P2;
    end
%     plot(path(:,1),path(:,2),'LineWidth',2)

%     plot(path(:,1),path(:,2),'ro');

%     angle = a1;
%     angles = [angles; angle];
% 
%     for j=1:length(path)-1
%         angle0 = angle;
%         C = ClothoidCurve();
%         x1=path(j,1);
%         y1=path(j,2);
%         x2=path(j+1,1);
%         y2=path(j+1,2);
%         angle1 = atan2d((y2 - y1),(x2 - x1))/180*pi;
%         if j+2 <= length(path)
%             x3=path(j+2,1);
%             y3=path(j+2,2);
%             angle2 = atan2d((y3 - y2),(x3 - x2))/180*pi;
%             angle = (angle1+angle2)/2;
%         else
%             angle = a2;
%         end
%         iter = C.build_G1(x1, y1, angle0, x2, y2, angle );
%         angles = [angles; angle];
%         C.plot( 1000, '-b', 'LineWidth', 3 );
% 
%     end

    npts = 1000;
    CL = ClothoidSplineG2();
    SL = CL.buildP1(path(:,1), path(:,2),a1,a2);
    SL.plot(npts,{'Color','blue','LineWidth',2},{'Color','blue','LineWidth',2});

    [a b] = SL.points(100);
    samples = [samples; a' b'];

    fmt = [repmat('%10.5f ', 1, length(samples)-1), '%10.5f\n'];
    fprintf(fid, fmt, samples);

    plot(P1(1), P1(2), 'ro', ...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','g',...
        'MarkerSize',5);

    plot(P2(1), P2(2), 'bo', ...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','y',...
        'MarkerSize',5);

    quiver( P1(1), P1(2), LVEC*cos(a1), LVEC*sin(a1), 'Color', 'r','LineWidth',1 );
    quiver( P2(1), P2(2), LVEC*cos(a2), LVEC*sin(a2), 'Color', 'r','LineWidth',1 );
end

fclose(fid);