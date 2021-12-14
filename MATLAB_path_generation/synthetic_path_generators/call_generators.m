function samples = call_generators(generator, map_name, num_traj, num_points, options)

res = 1;

if contains(generator, '_map') && ~contains(generator, '_void')
    if isempty(map_name)
        load exampleMaps.mat;
        map = binaryOccupancyMap(simpleMap,res);
    else
        image = imread([map_name,'.png']);
        grayimage = rgb2gray(image);
        bwimage = grayimage < 0.5;
        im = imresize(bwimage, 1/40);
        map = binaryOccupancyMap(im,res);
    end
elseif contains(generator, '_void') && ~contains(generator, '_map')
    image = imread('voidMap.png');
    grayimage = rgb2gray(image);
    bwimage = grayimage < 0.5;
    im = imresize(bwimage, 1/40);
    map = binaryOccupancyMap(im,res);
end

% Plot figure
figure(100);
show(map);
hold on;

% Get positions
[pos.x1, pos.y1] = ginput(1);
plot(pos.x1, pos.y1, 'xk', 'LineStyle', 'none');
drawnow;

[pos.x2, pos.y2] = ginput(1);
plot(pos.x2, pos.y2, 'xr', 'LineStyle', 'none');
drawnow;

map_res = copy(map);
inflate(map_res,0.2/res);

% Call generator
samples = feval(generator, num_traj, num_points, pos, map_res, res, options);

end