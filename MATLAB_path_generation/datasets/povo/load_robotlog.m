function [human, loc, odom, path, clusters, lidar, opti] = load_robotlog(folder, file)

% load_robotlog
% Alessandro Antonucci @AlexRookie
% University of Trento

human = [];
loc = [];
odom = [];
path = [];
clusters = [];
lidar = [];
opti = [];

%% Load human data

filename = [folder, '/hp_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        human.t(k,1)     = parsed_line.ts;
        human.t_rcv(k,1) = parsed_line.rcv_ts;
        human.pos(k,:)   = [parsed_line.x, parsed_line.y];
        human.valid(k,:) = parsed_line.valid;
        k = k+1;
    end
    fclose(fid);
end

%% Load loc data

filename = [folder, '/loc_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    loc = [];
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        loc.t_rcv(k,1) = parsed_line.rcv_ts;
        loc.pose(k,:)  = [parsed_line.loc_data.x, parsed_line.loc_data.y, parsed_line.loc_data.theta];
        k = k+1;
    end
    fclose(fid);
end

%% Load odom data

filename = [folder, '/odom_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    odom = [];
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        odom.t(k,1)     = parsed_line.ts;
        odom.t_rcv(k,1) = parsed_line.rcv_ts;
        odom.pose(k,:)  = [parsed_line.x, parsed_line.y, parsed_line.theta];
        odom.v(k,:)     = parsed_line.v;
        odom.omega(k,:) = parsed_line.omega;
        k = k+1;
    end
    fclose(fid);
end

%% Load path data

filename = [folder, '/path_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    path = [];
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        path.t(k,1)     = parsed_line.ts;
        path.t_rcv(k,1) = parsed_line.rcv_ts;
        if not(isempty(parsed_line.path))
            path.x0{k,1} = [parsed_line.path.x0];
            path.y0{k,1} = [parsed_line.path.y0];
            path.t0{k,1} = [parsed_line.path.t0];
            path.k0{k,1} = [parsed_line.path.k0];
            path.dk{k,1} = [parsed_line.path.dk];
            path.L{k,1}  = [parsed_line.path.L];
            assert(numel(path.x0(k,1))==numel(path.y0(k,1)), ...
                'Lenghts are different (x0:%d, y0:%d).', numel(path.x0(k,1)), numel(path.y0(k,1)));
        else
            path.x0{k,1} = [];
            path.y0{k,1} = [];
            path.t0{k,1} = [];
            path.k0{k,1} = [];
            path.dk{k,1} = [];
            path.L{k,1}  = [];
        end
        k = k+1;
    end
    fclose(fid);
end

%% Load lidar clusters data

filename = [folder, '/tld_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        clusters.t(k,1)     = parsed_line.ts;
        clusters.t_rcv(k,1) = parsed_line.rcv_ts;
        if not(isempty(parsed_line.clusters))
            clusters.centroids{k,1} = [parsed_line.clusters.x; parsed_line.clusters.y];
            for i = 1:size(clusters.centroids{k,1},2)
                clusters.points{k,i} = [parsed_line.clusters(i).points_x, parsed_line.clusters(i).points_y];
            end
            clusters.id{k,1}      = [parsed_line.clusters.id];
            clusters.type{k,1}    = [parsed_line.clusters.type];
            clusters.visible{k,1} = [parsed_line.clusters.visible];
        end
        k = k+1;
    end
    fclose(fid);
end

%% Load lidar data

% Precompute LIDAR data angles
VECTOR_SIZE = 360*2;
OFFSET_DEGS = 0.0;
scan_angle = (1:VECTOR_SIZE)*0.5 + OFFSET_DEGS;

filename = [folder, '/lidar_', file, '.txt'];
if exist(filename, 'file')
    fid = fopen(filename,'r');
    k = 1;
    while true
        line = fgetl(fid);
        if not(ischar(line))
            break;
        end
        if length(line) < 1
            error('Error while parsing data.');
        end
        parsed_line = jsondecode(line);
        lidar.t_rcv(k,1) = parsed_line.rcv_ts;
        lidar.size(k,1) = parsed_line.size;
        if not(isempty(parsed_line.data))
            assert(lidar.size(k,1) == VECTOR_SIZE, 'Expected %d angles, got %d.', VECTOR_SIZE, lidar.size(k,1));
            scan_dist = parsed_line.data;
            cart_x = NaN(VECTOR_SIZE,1);
            cart_y = NaN(VECTOR_SIZE,1);
            for i = 1:lidar.size(k,1)
                if (scan_dist == 0.0)
                    continue;
                end
                cart_x(i,1) = scan_dist(i) .* cos(scan_angle(i)*pi/180.0);
                cart_y(i,1) = scan_dist(i) .* sin(scan_angle(i)*pi/180.0);
            end
            lidar.points{k,1} = [cart_x, cart_y];
        end
        k = k+1;
    end
    fclose(fid);
end

%% Load optitrack data

% TDB

end
