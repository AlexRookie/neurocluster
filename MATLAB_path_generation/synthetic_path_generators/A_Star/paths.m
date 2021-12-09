res = 100;

map = binaryOccupancyMap(10,10,res);
x = [5.0];
y = [5.0];

setOccupancy(map, [x y],1);
inflate(map, 1);
show(map)
hold on

map1 = copy(map);
inflate(map1,0.3);

for i=1:10
    startLocation = [2*res 2*res];
    startLocation = [startLocation(1) + floor(rand()*res), startLocation(2) + floor(rand()*res)];
    endLocation = [8*res 8*res];
    endLocation = [endLocation(1) + floor(rand()*res), endLocation(2) + floor(rand()*res)];
    
    planner = plannerAStarGrid(map1);
    path = plan(planner,startLocation,endLocation);
    plot(path(:,1)/res,path(:,2)/res,'LineWidth',2)
end