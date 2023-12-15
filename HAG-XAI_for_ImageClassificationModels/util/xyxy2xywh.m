function p_new = xyxy2xywh(p)
    p_new = zeros(size(p),"like",p);
    p_new(:,1) = (p(:,1)+p(:,3))./2;
    p_new(:,2) = (p(:,2)+p(:,4))./2;
    p_new(:,3) = p(:,3)-p(:,1);
    p_new(:,4) = p(:,4)-p(:,2);

end