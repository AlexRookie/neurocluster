function Grid = createGrid(gridsize, Walls, maplims)

% Create grid map as polyshapes.
%
% In:
%   gridsize: cells dimension
%   Walls:    wall polyshapes
%   maplims:  map limits
%
% Out:
%   Grid.poly: grid polyshapes (N,M)
%   Grid.stat: grid static boolean matrix (-1 if grid cell intersect walls) (N,M)
%   Grid.size: grid cells size
%   Grid.bbox: grid cell bounding boxes (N*M,4)
%   Grid.cent: centroid positions (N*M,2)
%
% Alessandro Antonucci @AlexRookie
% University of Trento

TOL = 40; % percentage of area for intersection==false

xLim = maplims(1:2);
yLim = maplims(3:4);

% Grid cells boundaries
xgrid = round(xLim(1)):gridsize:round(xLim(2));
ygrid = round(yLim(1)):gridsize:round(yLim(2));

% Grid in polyshapes
for i = 1:length(xgrid)-1
    for j = 1:length(ygrid)-1
        Grid.poly(i,j) = polyshape([xgrid(i), xgrid(i+1), xgrid(i+1), xgrid(i)], ...
            [ygrid(j), ygrid(j), ygrid(j+1), ygrid(j+1)]);
    end
end

% Check wall intersections with grid map
Grid.stat = zeros(size(Grid.poly));
maxAreaOcc = area(Grid.poly(1))*(1-(TOL/100));
if not(isempty(Walls))
    for i = 1:size(Grid.poly,1)
        for j = 1:size(Grid.poly,2)
            
            % Find grid/wall intersections
            interPoly = intersect(Walls, Grid.poly(i,j));
            
            if not(isempty(interPoly.Vertices))
                if area(interPoly) < maxAreaOcc
                    % Update cell
                    Grid.poly(i,j) = subtract(Grid.poly(i,j), interPoly);
                else
                    % Set stat to -1
                    Grid.stat(i,j) = -1;
                end
            end
            
        end
    end
end

% Grid size
Grid.size = gridsize;

% Grid cell bounding box [x_g,y_g,width_g,height_g]
Grid.bbox = NaN(numel(Grid.poly),4);
for i = 1:numel(Grid.poly)
    Grid.bbox(i,:) = [min(Grid.poly(i).Vertices(:,1)), min(Grid.poly(i).Vertices(:,2)), gridsize, gridsize];
end

% Find cell centroids
Grid.cent = NaN(numel(Grid.poly),2);
for i = 1:numel(Grid.poly)
    if Grid.stat(i) == -1
        continue;
    end
    
    [cx, cy] = centroid(Grid.poly(i));
    Grid.cent(i,:) = [cx, cy];
end

end