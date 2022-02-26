it = 100;
map = 'povo2Atrium';                     % map name
positions = [1, 6, 0.0, 10.8, 1, -pi/2];

[Map, Pos] = map_and_positions(map, positions);

% Plot obstacle polyshapes and grid map
for i = 1:numel(Grid.poly)
    if Grid.theta(i) == 100
        continue;
    end
    quiver(Grid.cent(i,1), Grid.cent(i,2), 0.5*cos(Grid.theta(i)), 0.5*sin(Grid.theta(i)), 'color', 'b', 'linewidth', 2);
end

plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
for i = 1:numel(Grid.poly)
    if Grid.stat(i) == 0
        continue;
    end
    text(Grid.cent(i,1), Grid.cent(i,2), class_names{Grid.stat(i)}, 'color', '#77AC30', 'fontsize', 18, 'FontWeight', 'bold');
end