% function to define valid area where run montecarlo generator of points
function area = montecarlo_area(obstacles, pos)
    
    radius = 1;

    % Find a valid area for the starting point (1) and ending point (2)
    area.c1 = nsidedpoly(100, 'Center', [pos.x1 pos.y1], 'Radius', radius);
    for i = 1:length(obstacles)
        area_obs = obstacles{i,1}';
        if not(isinterior(area_obs, pos.x1, pos.y1))
            area.c1 = subtract(area.c1, obstacles{i,1}');
        else
            tmp = subtract(area.c1, obstacles{i,1}');
            area.c1 = subtract(area.c1, tmp);
        end
    end
    plot(area.c1, 'FaceColor', 'r');
    area.c2 = nsidedpoly(100, 'Center', [pos.x2 pos.y2], 'Radius', radius);
    for i = 1:length(obstacles)
        area_obs = obstacles{i,1}';
        if not(isinterior(area_obs, pos.x2, pos.y2))
            area.c2 = subtract(area.c2, obstacles{i,1}');
        else
            tmp = subtract(area.c2, obstacles{i,1}');
            area.c2 = subtract(area.c2, tmp);
        end
    end
    plot(area.c2, 'FaceColor', 'r');

end