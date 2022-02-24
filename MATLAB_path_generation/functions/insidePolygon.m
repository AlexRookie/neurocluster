function [in] = insidePolygon(polygon, point)

% Check if point is inside polygon via winding number method.
%
% In:
%   polygon: matrix of area vertices (2,n)
%   point:   point x-y
%
% Out:
%   in: boolean 1 if point is inside polygon
%
% Paolo Bevilacqua, Alessandro Antonucci
% University of Trento

% Check input shapes
if size(polygon,1) > size(polygon,2)
    polygon = polygon';
end

wm = 0; % winding number counter

% Close polygon
if norm(polygon(:,1)-polygon(:,end)) > 1e-5
    polygon(:,end+1) = polygon(:,1);
end

% Loop through all edges of the polygon
for i = 1:size(polygon,2)-1
    
    if polygon(2,i) <= point(2)
        % Start y <= P.y
        if polygon(2,i+1) > point(2) % an upward crossing
            if isLeft(polygon(:,i), polygon(:,i+1), point) > 0 % P left of edge
                wm = wm+1; % have a valid up intersect
            end
        end
        
    else
        
        % Start y > P.y (no test needed)
        if (polygon(2,i+1) <= point(2)) % a downward crossing
            if isLeft(polygon(:,i), polygon(:,i+1), point) < 0 % P right of edge
                wm = wm-1; % have a valid down intersect
            end
        end
    end
    
end

in = not(wm==0);

end

function left = isLeft(P0, P1, point)

left = (P1(1) - P0(1)) * (point(2) - P0(2)) - (point(1) - P0(1)) * (P1(2) - P0(2));

end