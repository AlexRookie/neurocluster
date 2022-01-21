function samples = call_generators(generator, map_name, num_traj, num_points, options)

res = 10;
obstacles = cell(0);

LVEC = 0.5;

% Load map
if not(isempty(map_name))
    % Open file
    fid = fopen([map_name,'.txt'],'r');
    if fid == -1
        error(['Impossible to open file: ', map_name]);
    end
    
    y_lim = [inf,-inf];
    x_lim = [inf,-inf];
    while ~feof(fid)
        line = fgets(fid);
        sz = sscanf(line, '%d');
        xV = [];
        yV = [];
        for i = 1:sz
            line = fgets(fid);
            pts = sscanf(line, '%f %f');
            % Bounding box
            xMin = min(xV);
            yMin = min(yV);
            xMax = max(xV);
            yMax = max(yV);
            if xMin < x_lim(1)
                x_lim(1) = xMin;
            end
            if yMin < y_lim(1)
                y_lim(1) = yMin;
            end
            if xMax > x_lim(2)
                x_lim(2) = xMax;
            end
            if yMax > y_lim(2)
                y_lim(2) = yMax;
            end
            % Vertex
            xV(end+1) = pts(1);
            yV(end+1) = pts(2);
        end
        % Obstacle vertices
        obstacles(end+1,1) = {[xV;yV]};
    end
    
    x_lim = [floor(x_lim(1)), floor(x_lim(2))];
    y_lim = [ceil(y_lim(1)), ceil(y_lim(2))];
    
    % Close file
    fclose(fid);
    
    % Plot map
    if options.plot
        fillFlag = true;
        figure(101);
        hold on, axis equal, grid on, axis tight, box on;
        for i = 1:size(obstacles,1)
           xV = obstacles{i,1}(1,:);
           yV = obstacles{i,1}(2,:);
           if fillFlag == 0
               plot(xV, yV, [0.7,0.7,0.65], 'linewidth', 1);
           else
               fill(xV,yV, [0.7,0.7,0.65], 'facealpha', 1);
           end
        end
    end
    
    % Shift map origin
    if x_lim(1) < 0
        for i = 1:size(obstacles,1)
            obstacles{i}(1,:) = obstacles{i}(1,:) + abs(x_lim(1));
        end
        x_lim = x_lim + abs(x_lim(1));
    end
    if x_lim(1) > 0
        for i = 1:size(obstacles,1)
            obstacles{i}(1,:) = obstacles{i}(1,:) - abs(x_lim(1));
        end
        x_lim = x_lim - abs(x_lim(1));
    end
    if y_lim(1) < 0
        for i = 1:size(obstacles,1)
            obstacles{i}(2,:) = obstacles{i}(2,:) + abs(y_lim(1));
        end
        y_lim = y_lim + abs(y_lim(1));
    end
    if y_lim(1) > 0
        for i = 1:size(obstacles,1)
            obstacles{i}(2,:) = obstacles{i}(2,:) - abs(y_lim(1));
        end
        y_lim = y_lim - abs(y_lim(1));
    end
    
    % Create occupancy map
    map = false((y_lim(2)-y_lim(1)).*res, (x_lim(2)-x_lim(1)).*res);
    for i = 1:size(obstacles,1)
        map = map | poly2mask(obstacles{i,1}(1,:).*res, obstacles{i,1}(2,:).*res, (y_lim(2)-y_lim(1)).*res, (x_lim(2)-x_lim(1)).*res);
    end
    map = flip(map);
    map = binaryOccupancyMap(map,res);
end

% Load example map
if isempty(map_name)
    load exampleMaps.mat;
    map = binaryOccupancyMap(simpleMap,res);
end

% image = imread([map_name,'.png']);
% grayimage = rgb2gray(image);
% bwimage = grayimage < 0.5;
% im = imresize(bwimage, 1/40);
% map = binaryOccupancyMap(im,res);

% Plot figure
figure(100);
show(map);
xlim(x_lim);
ylim(y_lim);
hold on;

% Get position 1
[obj_pos.x1, obj_pos.y1] = ginput(1);
while  checkOccupancy(map,[obj_pos.x1, obj_pos.y1])
    [obj_pos.x1, obj_pos.y1] = ginput(1);
end
plot(obj_pos.x1, obj_pos.y1, 'xk', 'LineStyle', 'none');
drawnow;
% Get angle 1
[x, y] = ginput(1);
obj_pos.a1 = atan2((y - obj_pos.y1), (x - obj_pos.x1));
quiver( obj_pos.x1, obj_pos.y1, LVEC*cos(obj_pos.a1), LVEC*sin(obj_pos.a1), 'Color', 'r' );

% Get position 2
[obj_pos.x2, obj_pos.y2] = ginput(1);
while  checkOccupancy(map,[obj_pos.x2, obj_pos.y2])
    [obj_pos.x2, obj_pos.y2] = ginput(1);
end
plot(obj_pos.x2, obj_pos.y2, 'xr', 'LineStyle', 'none');
drawnow;
% Get angle 2
[x, y] = ginput(1);
obj_pos.a2 = atan2((y - obj_pos.y2), (x - obj_pos.x2));
quiver( obj_pos.x2, obj_pos.y2, LVEC*cos(obj_pos.a2), LVEC*sin(obj_pos.a2), 'Color', 'r' );

% Inflate occupancy map
map_res = copy(map);
inflate(map_res, 0.5);

% Get polygons of inflated map and set as obstacles
obstacles = map2poly(map_res,res);

obj_map.obstacles = obstacles;
obj_map.res = res;
obj_map.map_res = map_res;

% Call generator
samples = feval(generator, num_traj, num_points, obj_pos, obj_map, options);

end