function pout = homography2d_project(H, pin)
    % homography_project(H, pin) takes the 3x3 homogeneous matrix H and a 2xn
    % matrix of input points, and returns a 2xn matrix of transformed output
    % points
    %   H    homography matrix
    %   pin  pixels in the camera view

    if not(size(pin,1) == 2)
        pin = pin';
    end

    n = size(pin, 2);

    q = H * [pin; ones(1, n)];
    lambda = q(3,:);
    pout = [q(1,:)./lambda; q(2,:)./lambda]';
end