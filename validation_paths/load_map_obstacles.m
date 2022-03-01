function [obstacles, x_lim, y_lim, map] = load_map_obstacles(mapname)
    %  loadMapObstacles(Map)
    %  Loads the scenario walls expressed as polygons.
    %    Map: struct with map name and scale
    
    file = [mapname,'.txt'];
    
    obstacles = {};
    
    % Opens file
    fid = fopen(file,'r');
    
    if fid == -1
        error(['Impossible to open the file: ',file]);
    end
    
    y_lim = [inf,-inf];
    x_lim = [inf,-inf];
    
    % figure; hold on; axis equal;
    while ~feof(fid)
         line = fgets(fid);
         sz = sscanf(line, '%d');
         xBound = [];
         yBound = [];
         xV = [];
         yV = [];
         
         for i = 1:sz
             line = fgets(fid);
             pts = sscanf(line, '%f %f');
             
             % Creates bounding box
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
             
             xBound = [xMin,xMin,xMax,xMax,xMin];         
             yBound = [yMin,yMax,yMax,yMin,yMin];         
             xV(end+1) = pts(1);
             yV(end+1) = pts(2);
         end
         
         Pmean = [mean(xBound(1:end-1));mean(yBound(1:end-1))];
         
         obstacles(end+1,1) = {Pmean};         % Obstacle x and y middle point
         obstacles(end,2) = {[xV;yV]};         % Obstacle points
         obstacles(end,3) = {[xBound;yBound]}; % Obstacle maximum vertexes
    end
    
    fclose(fid);

    res = 10;

    % Create occupancy map
    map = false(round((y_lim(2)-y_lim(1)).*res), round((x_lim(2)-x_lim(1)).*res));
    for i = 1:size(obstacles,1)
        map = map | poly2mask((obstacles{i,2}(1,:)-x_lim(1)).*res, (obstacles{i,2}(2,:)-y_lim(1)).*res, round((y_lim(2)-y_lim(1)).*res), round((x_lim(2)-x_lim(1)).*res));
    end
    map = flipud(map);
    map = binaryOccupancyMap(map,res);

end
