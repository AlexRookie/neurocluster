function [Map, Pos] = map_and_positions(varargin) % map_and_positions(map_name, [positions], [i])

map_name = varargin{1};
if (nargin >= 2)
    positions = varargin{2};
else
    positions = [];
end

if (nargin >= 3)
    it = varargin{3};
else
    it = 0;
end

res = 10;        % occupancy map resolution
inflation = 0.1; % obstacles inflation (meters)

l_vec = 0.5; % orientation angle length (for plot)

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
    
    % DEBUG: plot obstacle map
    %{
    figure(120);
    fillFlag = true;
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
    %}
   
    % Shift map origin to (0,0)
    trasl_x = 0;
    trasl_y = 0;
    if x_lim(1) < 0
        trasl_x = abs(x_lim(1));
    elseif x_lim(1) > 0
        trasl_x = - abs(x_lim(1));
    end
    if y_lim(1) < 0
        trasl_y = abs(y_lim(1));
    elseif x_lim(1) > 0
        trasl_y = - abs(y_lim(1));
    end
    for i = 1:size(obstacles,1)
        obstacles{i}(1,:) = obstacles{i}(1,:) + trasl_x;
        obstacles{i}(2,:) = obstacles{i}(2,:) + trasl_y;
    end
    x_lim = x_lim + trasl_x;
    y_lim = y_lim + trasl_y;
    
    % Create occupancy map
    map = false((y_lim(2)-y_lim(1)).*res, (x_lim(2)-x_lim(1)).*res);
    for i = 1:size(obstacles,1)
        map = map | poly2mask(obstacles{i,1}(1,:).*res, obstacles{i,1}(2,:).*res, (y_lim(2)-y_lim(1)).*res, (x_lim(2)-x_lim(1)).*res);
    end
    map = flipud(map);
    map = binaryOccupancyMap(map,res);
end

% Load example map
if isempty(map_name)
    load exampleMaps.mat;
    map = binaryOccupancyMap(simpleMap,res);
end

% Plot figure
figure(100+it);
show(map);
xlim(x_lim);
ylim(y_lim);
hold on;

if strcmp(map_name, 'povo2Atrium')
    r = readlines('povo2Atrium_listaQR.txt');
    qr = jsondecode(r{1}).value;
    hold on;
    for i = 1:length(qr)
        plot(qr(i).x + trasl_x, qr(i).y + trasl_y, '*');
        text(qr(i).x + trasl_x, qr(i).y + trasl_y, num2str(qr(i).id));
    end
end

if isempty(positions)    
    % Get position 1
    [Pos.x1, Pos.y1] = ginput(1);
    while  checkOccupancy(map,[Pos.x1, Pos.y1])
        [Pos.x1, Pos.y1] = ginput(1);
    end
    plot(Pos.x1, Pos.y1, 'xk', 'LineStyle', 'none');
    drawnow;
    % Get angle 1
    [x, y] = ginput(1);
    Pos.a1 = atan2((y - Pos.y1), (x - Pos.x1));
    quiver( Pos.x1, Pos.y1, l_vec*cos(Pos.a1), l_vec*sin(Pos.a1), 'Color', 'r' );
    drawnow;
    % Get position 2
    [Pos.x2, Pos.y2] = ginput(1);
    while  checkOccupancy(map,[Pos.x2, Pos.y2])
        [Pos.x2, Pos.y2] = ginput(1);
    end
    plot(Pos.x2, Pos.y2, 'xr', 'LineStyle', 'none');
    drawnow;
    % Get angle 2
    [x, y] = ginput(1);
    Pos.a2 = atan2((y - Pos.y2), (x - Pos.x2));
    quiver( Pos.x2, Pos.y2, l_vec*cos(Pos.a2), l_vec*sin(Pos.a2), 'Color', 'r' );
    drawnow;
else
    % Read position and angle 1
    Pos.x1 = positions(1);
    Pos.y1 = positions(2);
    Pos.a1 = positions(3);
    % Read position and angle 2
    Pos.x2 = positions(4);
    Pos.y2 = positions(5);
    Pos.a2 = positions(6);

    plot(Pos.x1, Pos.y1, 'xk', 'LineStyle', 'none');
    quiver( Pos.x1, Pos.y1, l_vec*cos(Pos.a1), l_vec*sin(Pos.a1), 'Color', 'r' );
    plot(Pos.x2, Pos.y2, 'xr', 'LineStyle', 'none');
    quiver( Pos.x2, Pos.y2, l_vec*cos(Pos.a2), l_vec*sin(Pos.a2), 'Color', 'r' );
    drawnow;
end

% Inflate occupancy map
map_res = copy(map);
inflate(map_res, inflation);

% Get polygons of inflated map and set as obstacles
poly_obstacles = map2poly(map_res, res);
% DEBUG: plot polygons
%{
cellfun(@plot, poly_obstacles);
axis([-1,20,-1,20]);
%}

% Find a valid area to generate starting/ending points
area = montecarlo_area(poly_obstacles, Pos);

Map.area = area;
Map.obstacles = obstacles;
Map.poly_obstacles = poly_obstacles;
Map.res = res;
Map.map_res = map_res;

end