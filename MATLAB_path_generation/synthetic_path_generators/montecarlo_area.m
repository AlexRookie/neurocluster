function area = montecarlo_area(map, pos)
    
    [area.x1, area.y1] = valid_area(map, pos.x1, pos.y1);
    [area.x2, area.y2] = valid_area(map, pos.x2, pos.y2);

    area.x1(1) = area.x1(1) + 0.1;
    area.y1(1) = area.y1(1) + 0.1;
    area.x2(1) = area.x2(1) + 0.1;
    area.y2(1) = area.y2(1) + 0.1;
    area.x1(2) = area.x1(2) - 0.1;
    area.y1(2) = area.y1(2) - 0.1;
    area.x2(2) = area.x2(2) - 0.1;
    area.y2(2) = area.y2(2) - 0.1;

end

function [x, y] = valid_area(map, a, b)

    occ = getOccupancy(map);
    mask = occ*0;

    x = max(floor(a)-1,0);
    y = max(floor(b)-1,0);
    x = [x, min(ceil(a),map.DataSize(1)-1)];
    y = [y, min(ceil(b),map.DataSize(2)-1)];
    
    while x(2) - x(1) ~= y(2) - y(1)
        if x(2) - x(1) < y(2) - y(1)
            if x(1) > map.DataSize(1)/2
                x(1) = x(1)-1;
            else
                x(2) = x(2)+1;
            end
        elseif x(2) - x(1) > y(2) - y(1)
            if y(1) > map.DataSize(2)/2
                y(1) = y(1)-1;
            else
                y(2) = y(2)+1;
            end
        end
    end

    n = x(2) - x(1) + 1;

    xx = (x(1):x(2))+1;
    yy = (y(1):y(2))+1;

    mask(yy,xx) = ones(n,n);
    [x,y,coll] = get_boundaries(mask, occ);
    
    flag = true;
    while flag
        [flag, mask, xx, yy] = checker(mask, coll, xx, yy);
        [x,y,coll] = get_boundaries(mask, occ);
    end

end

function [x,y,coll] = get_boundaries(mask, occ)

    coll = max(xor(mask, occ) - occ,0);

    [v,i]=max(coll);
    pos = i .* (v & i);
    y(1) = max(pos)-1;
    count_x = sum(pos~=0);
    [v,i]=max(coll');
    pos = i .* (v & i);
    x(1) = max(pos)-1;
    count_y = sum(pos~=0);
    x(2) = x(1) + count_x;
    y(2) = y(1) + count_y;

end

function [flag, mask, xx, yy] = checker(mask, coll, xx, yy)
    check = coll(yy,xx)
    flag = false;
    if sum(check(:) == 0) > 0
        if check(1,1) == 0
            if sum(check(1,:)) < 2 && sum(check(:,1)) < 2
                flag = true;
                [n,m] = size(check);
                check = [check; ones(1,m)];
                [n,m] = size(check);
                check = [check, ones(n,1)];
                xx = [xx, xx(end)+1];
                yy = [yy, yy(end)+1];
                check(1,:) = 0;
                check(:,1) = 0;
                mask(yy,xx) = check;
                xx(1) = [];
                yy(1) = [];
            else
                check(1,:) = 0;
                check(:,1) = 0;
                mask(yy,xx) = check;
            end
        elseif check(1,end) == 0
            if sum(check(1,:)) < 2 && sum(check(:,end)) < 2
                flag = true;
                [n,m] = size(check);
                check = [check; ones(1,m)];
                [n,m] = size(check);
                check = [ones(n,1), check];
                xx = [xx(1)-1, xx];
                yy = [yy, yy(end)+1];
                check(1,:) = 0;
                check(:,end) = 0;
                mask(yy,xx) = check;
                xx(end) = [];
                yy(1) = [];
            else
                check(1,:) = 0;
                check(:,end) = 0;
                mask(yy,xx) = check;
            end
        elseif check(end,1) == 0
            if sum(check(end,:)) < 2 && sum(check(:,1)) < 2
                flag = true;
                [n,m] = size(check);
                check = [ones(1,m); check];
                [n,m] = size(check);
                check = [check, ones(n,1)];
                xx = [xx, xx(end)+1];
                yy = [yy(1)-1, yy];
                check(:,1) = 0;
                check(end,:) = 0;
                mask(yy,xx) = check;
                xx(1) = [];
                yy(end) = [];
            else
                check(:,1) = 0;
                check(end,:) = 0;
                mask(yy,xx) = check;
            end
        elseif check(end,end) == 0
            if sum(check(end,:)) < 2 && sum(check(:,end)) < 2
                flag = true;
                [n,m] = size(check);
                check = [ones(1,m); check];
                [n,m] = size(check);
                check = [ones(n,1), check];
                xx = [xx(1)-1, xx];
                yy = [yy(1)-1, yy];
                check(end,:) = 0;
                check(:,end) = 0;
                mask(yy,xx) = check;
                xx(end) = [];
                yy(end) = [];
            else
                check(end,:) = 0;
                check(:,end) = 0;
                mask(yy,xx) = check;
            end
        end
    end
end