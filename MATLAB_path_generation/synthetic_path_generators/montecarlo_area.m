% function to define valid area where run montecarlo generator of points
function area = montecarlo_area(obstacles, pos)
    
    % Find a valid area for the starting point (1) and ending point (2)
    area.c1 = nsidedpoly(100, 'Center', [pos.x1 pos.y1], 'Radius', 1);
    for i = 1:length(obstacles)
        area.c1 = subtract(area.c1,polyshape(obstacles{i,1}'));
    end
    plot(area.c1, 'FaceColor', 'r');
    area.c2 = nsidedpoly(100, 'Center', [pos.x2 pos.y2], 'Radius', 1);
    for i = 1:length(obstacles)
        area.c2 = subtract(area.c2,polyshape(obstacles{i,1}'));
    end
    plot(area.c2, 'FaceColor', 'r');

end