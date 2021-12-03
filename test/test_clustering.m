clear all;

n = 30;

a = [1:100; 1:100];
b = permute(repmat(a,1,1,n), [3,1,2]);
T1 = b + rand(size(b));
a = [1:100; 100:-1:1];
b = permute(repmat(a,1,1,n), [3,1,2]);
T2 = b + rand(size(b));
a = [-1:-1:-100; 100:-1:1];
b = permute(repmat(a,1,1,n), [3,1,2]);
T3 = b + rand(size(b));

T = vertcat([T1;T2;T3]);

close all;
hold on;
for i = 1:size(T,1)
    plot(squeeze(T(i,1,:)), squeeze(T(i,2,:)));
end

Tr = reshape(T,size(T,1),[]);

% compute the mean of the dataset along the columns (features, pixels)
meanT = mean(Tr,1);

% compute the principal components (by default the algorithm remove the mean from the data)
[pcs,scores,~,~,explained] = pca(Tr-meanT);

% Pareto plot to visualize the explained variance of the principal scores (eigenvalues)
figure;
pareto(explained);
title('\textbf{Pareto Plot of the Explained Variance}','FontSize',16,'Interpreter','latex');
ylabel('Cumulative Explained Variance','Interpreter','latex','FontSize',14);
xlabel('\# Components','Interpreter','latex','FontSize',14);

figure;
subplot(131);
a = scores(1,1)*pcs(:,1)';
b = reshape(a,2,100);
plot(b(1,:),b(2,:));
title('\textbf{1st pcomp}','FontSize',14,'Interpreter','latex');
subplot(132);
a = scores(1,2)*pcs(:,2)';
b = reshape(a,2,100);
plot(b(1,:),b(2,:));
title('\textbf{2nd pcomp}','FontSize',14,'Interpreter','latex');
subplot(133);
a = scores(1,3)*pcs(:,3)';
b = reshape(a,2,100);
plot(b(1,:),b(2,:));
title('\textbf{3rd pcomp}','FontSize',14,'Interpreter','latex');
