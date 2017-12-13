function L_f = liner_interp(x, XX, YY),
    interp_point_num = length(YY);
    seg_len = floor(length(x) / (interp_point_num-1));
    L_f = zeros(1, length(x));
    for i = 1:(interp_point_num-1),
        xx = [XX(i) XX(i+1)];
        yy = [YY(i) YY(i+1)];
        seg_start = (i-1)*seg_len + 1;
        seg_end = i*seg_len;
        L_f(1, seg_start:seg_end) = lagrange_lu(x(1, seg_start:seg_end), xx, yy, 1);
    end
end
