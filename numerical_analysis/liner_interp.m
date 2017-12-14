function Li_x = liner_interp(X, XX, YY)
    interp_point_num = length(YY);
    seg_len = floor(length(X) / (interp_point_num-1));
    Li_x = zeros(1, length(X));
    for i = 1:(interp_point_num-1),
        xx = [XX(i) XX(i+1)];
        yy = [YY(i) YY(i+1)];
        seg_start = (i-1)*seg_len + 1;
        seg_end = i*seg_len;
        Li_x(1, seg_start:seg_end) = lagrange_lu(X(1, seg_start:seg_end), xx, yy, 1);
    end
end
