function obstacles = map2poly(map,res)

obstacles = cell(0);

% Get contour from binary occupancy map
cmatrix = bwboundaries(rot90(getOccupancy(map),-1));

% Rescale the polynomials to real dimensions of map
for i = 1:length(cmatrix)
    dim = map.GridSize(1);
    cmatrix{i}(:,1) = ((cmatrix{i}(:,1)-1)./(dim-1).*dim)./res;
    dim = map.GridSize(2);
    cmatrix{i}(:,2) = ((cmatrix{i}(:,2)-1)./(dim-1).*dim)./res;
end

% Create obstacle polynomials
for i = 1:length(cmatrix)
    polyob = polyshape(cmatrix{i});
    obstacles{end+1,1} = polybuffer(polyob, 1/res, 'jointtype', 'round');
end

end