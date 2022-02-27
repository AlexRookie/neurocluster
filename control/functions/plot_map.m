function h = plot_map(map_name, plot_qr)
% Alessandro Antonucci @AlexRookie
% University of Trento

h = [];
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
end

trasl_x = 0;
trasl_y = 0;

% Shift map origin to (0,0)
%{
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
%}

fillFlag = true;
for i = 1:size(obstacles,1)
    xV = obstacles{i,1}(1,:);
    yV = obstacles{i,1}(2,:);
    if fillFlag == 0
        h(end+1) = plot(xV, yV, [0.7,0.7,0.65], 'linewidth', 1);
    else
        h(end+1) = fill(xV,yV, [0.7,0.7,0.65], 'facealpha', 1);
    end
end

if strcmp(map_name, 'povo2Atrium') && (plot_qr == true)
    r = readlines('povo2Atrium_listaQR.txt');
    qr = jsondecode(r{1}).value;
    hold on;
    for i = 1:length(qr)
        h(end+1) = plot(qr(i).x + trasl_x, qr(i).y + trasl_y, '*');
        h(end+1) = text(qr(i).x + trasl_x, qr(i).y + trasl_y, num2str(qr(i).id));
    end
end

end