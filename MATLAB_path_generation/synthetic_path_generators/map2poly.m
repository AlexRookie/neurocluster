function obstacles = map2poly(map,res)

    obstacles = cell(0);

    % Get contour from binary occupancy map
    figure(200);
    cmatrix = contour(getOccupancy(map));
    close(200);

    [x, y, z] = C2xyz(cmatrix);
    shp = struct('Geometry', 'PolyLine', 'X', x, 'Y', y, 'Z', num2cell(z));

    % Rescale the polynomials to real dimensions of map
    for i=1:length(shp)
        dim = map.DataSize(1);
        shp(i).X = ((shp(i).X-1)./(dim-1).*dim)./res;
        dim = map.DataSize(2);
        shp(i).Y = ((shp(i).Y-1)./(dim-1).*dim)./res;
    end
    % Remove duplicate points in polynomial
    for i = 1:size(obstacles,1)
        j = 1;
        while j < size(obstacles{i,1},2)
            if obstacles{i,1}(:,j) == obstacles{i,1}(:,j+1)
                obstacles{i,1}(:,j+1) = [];
            else
                j = j + 1;
            end
        end
    end

    % Save obstacle polynomials in array
    for i=1:length(shp)
        if shp(i).Z == 0.5
            % Plot the inflated map polygons
            plot(shp(i).X,shp(i).Y,'r');
            obstacles(end+1,1) = {[shp(i).X;shp(i).Y]};
        end
    end

end