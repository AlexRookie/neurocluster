function samples = call_generators(generator, map_name, num_traj, num_points, options)

res = 1;
obstacles = cell(0);

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
        %fillFlag = true;
        %figure(101);
        %hold on, axis equal, grid on, axis tight, box on;
        %for i = 1:size(obstacles,1)
        %    xV = obstacles{i,1}(1,:);
        %    yV = obstacles{i,1}(2,:);
        %    if fillFlag == 0
        %        plot(xV, yV, [0.7,0.7,0.65], 'linewidth', 1);
        %    else
        %        fill(xV,yV, [0.7,0.7,0.65], 'facealpha', 1);
        %    end
        %end
    end
    
    % Create occupancy map
    map = false(x_lim(2)-x_lim(1), x_lim(2)-x_lim(1));
    for i = 1:size(obstacles,1)
        map = map | poly2mask(obstacles{i,1}(1,:), obstacles{i,1}(2,:), x_lim(2)-x_lim(1), x_lim(2)-x_lim(1));
    end
    map = flip(map);
    map = binaryOccupancyMap(map, res);
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
hold on;

% Get positions
[pos.x1, pos.y1] = ginput(1);
while  checkOccupancy(map,[pos.x1, pos.y1])
    [pos.x1, pos.y1] = ginput(1);
end
plot(pos.x1, pos.y1, 'xk', 'LineStyle', 'none');
drawnow;

[pos.x2, pos.y2] = ginput(1);
while  checkOccupancy(map,[pos.x2, pos.y2])
    [pos.x2, pos.y2] = ginput(1);
end
plot(pos.x2, pos.y2, 'xr', 'LineStyle', 'none');
drawnow;

% Inflate occupancy map
map_res = copy(map);
inflate(map_res, 0.2/res);

ob_map.obstacles = obstacles;
ob_map.res = res;
ob_map.map_res = map_res;

% Call generator
samples = feval(generator, num_traj, num_points, pos, ob_map, options);

end