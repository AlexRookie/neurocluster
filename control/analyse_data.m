class_names = {'L', 'R', 'S'};

figure(12);
hold on, axis equal, grid on;

for i = 1:out.x.Length
    if (out.valid.Data(i))
        plot(out.x.Data(i,end), out.y.Data(i,end), '*');
         text(out.x.Data(i,end)+0.2, out.y.Data(i,end)+0.4, [class_names{out.class.Data(i)}, ' ', num2str(round(out.conf.Data(i,out.class.Data(i))*100))], 'color', 'r', 'fontsize', 16);
    end
end