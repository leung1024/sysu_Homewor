function L_f = liner_interp(x, xx, yy),
    interp_point_num = length(yy);
    seg_len = floor(length(x) / (interp_point_num-1));
    L_f = zeros(1, length(x));
    for i = 1:(interp_point_num-1),
        XX = [xx(i) xx(i+1)];
        YY = [yy(i) yy(i+1)];
        seg_start = (i-1)*seg_len + 1;
        seg_end = i*seg_len;
        L_f(1, seg_start:seg_end) = lagrange_lu(x(1, seg_start:seg_end), XX, YY, 1);
    end
end
