function [h] = plot_map(walls, fillFlag, style)

% Draws the static obstacle map.
%
% In:
%   walls:    the obstacles struct
%   fillFlag: if 1 enables polygon's fillng
%   style:    colors and width/facealpha
%
% Out:
%   h: figure handle

h = [];

for i = 1:size(walls,1)
    
     xV = walls{i,1}(1,:);
     yV = walls{i,1}(2,:);
     
     if fillFlag == 0         
        h = [h, plot(xV, yV, style{1}, 'linewidth', style{2})];
     else
        h = [h, fill(xV,yV, style{1}, 'facealpha', style{2})];
     end
     
end

end
