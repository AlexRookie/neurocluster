function samples = load_dataset(dataset, num_points, options)

%step = 0.05; % sampling step (cm)

filename = ['./datasets/', dataset, '.mat'];
load(filename);

if options.plot
%     figure(100);
%     hold on, grid on, box on, axis equal;
%     axis(Data.AxisLim);
%     xlabel('x (m)');
%     xlabel('y (m)');
%     title('Dataset', 'interpreter', 'latex');
%     cellfun(@(x) plot(x(:,1), x(:,2), 'LineWidth', 2), Data.Humans);
end

x = cell(1,length(Data.Humans));
y = cell(1,length(Data.Humans));

for i = 1:length(Data.Humans)
    if length(Data.Humans{i}) < num_points
        continue;
    end
    x{i} = Data.Humans{i}(:,1)';
    y{i} = Data.Humans{i}(:,2)';
end

samples.x = x(~cellfun('isempty', x));
samples.y = y(~cellfun('isempty', y));

clear Data;

end