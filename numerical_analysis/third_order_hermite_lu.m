function h_3 = third_order_hermite_lu(X, XX, YY, Yd)
    x_len = length(X);
    interp_point_num = length(XX);
    seg_len = floor(length(X) / (interp_point_num-1));
    h_3 = zeros(1, x_len);
    for i = 1:(interp_point_num-1),
        seg_start = (i-1)*seg_len + 1;
        seg_end = i*seg_len;
        cur_X = X(1, seg_start:seg_end);
        a_1 = (1 + 2.*((cur_X - XX(i)) ./ (XX(i+1) - XX(i)))) .* ((cur_X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        a_2 = (1 + 2.*((cur_X - XX(i+1)) ./ (XX(i) - XX(i+1)))) .* ((cur_X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
        b_1 = (cur_X - XX(i)) .* ((cur_X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        b_2 = (cur_X - XX(i+1)) .* ((cur_X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
        h_3(1, seg_start:seg_end) = YY(i).*a_1 + YY(i+1).*a_2 + Yd(i).*b_1 + Yd(i+1).*b_2;
    end
end