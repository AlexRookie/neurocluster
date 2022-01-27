function conf = confusion_matrix(T, Y)

% Plot multiclass confusion matrix
% T: true samples
% Y: predicted samples

% Alessandro Antonucci @AlexRookie
% University of Trento

if size(T,1) > size(T,2)
    T = T';
end
if size(Y,1) > size(Y,2)
    Y = Y';
end
if any(T==0) || any(Y==0)
    T = T+1;
    Y = Y+1;
end

M = size(unique(T), 2);
N = size(T, 2);

targets = zeros(M,N);
outputs = zeros(M,N);
targetsIdx = sub2ind(size(targets), T, 1:N);
outputsIdx = sub2ind(size(outputs), Y, 1:N);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;

figure(300);
conf = plotconfusion(targets,outputs);

end