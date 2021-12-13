function samples = call_generators(generator, map_name, num_traj, num_points, options)

res = 1;

if contains(generator, '_map') && ~contains(generator, '_voidMap')
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
elseif contains(generator, '_voidMap') && ~contains(generator, '_map')
    image = imread('voidMap.png');
    grayimage = rgb2gray(image);
    bwimage = grayimage < 0.5;
    im = imresize(bwimage, 1/40);
    map = binaryOccupancyMap(im,res);
end

% Plot figure
if options.plot
    figure(100);
    show(map);
    hold on;
end

map1 = copy(map);
inflate(map1,0.2/res);

% Call generator
samples = feval(generator, num_traj, num_points, map1, res, options);

end