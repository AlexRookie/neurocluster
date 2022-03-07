function H = homography2d_solve(pin, pout)
    % homography2d_solve(pin, pout) takes a 2xn matrix of input points and
    % a 2xn matrix of output points, and returns the homogeneous
    % transformation matrix
    %   pin   pixels in the camera view
    %   pout  pixels in the new system

    if not(size(pin,1) == 2)
        pin = pin';
    end
    if not(size(pout,1) == 2)
        pout = pout';
    end

    n = size(pin, 2);
    if n < 4
        error('Error: need at least 4 matching points.');
    end

    j = 1;
    for i = 1:n
        x = pin(1,i); y = pin(2,i); X = pout(1,i); Y = pout(2,i);

        Ax = [-x -y -1 0 0 0 x.*X y.*X X];
        Ay = [0 0 0 -x -y -1 x.*Y y.*Y Y];

        A(j,:)   = Ax;
        A(j+1,:) = Ay;
        j = j+2;
    end

    [~, ~, V] = svd(A);
    H = (reshape(V(:,9), 3, 3)).';
end