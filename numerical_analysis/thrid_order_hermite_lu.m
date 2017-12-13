function h_3 = thrid_order_hermite_lu(X, XX, YY, Yd)
    x_len = length(X);
    interp_point_num = length(XX);
    h_3 = zeros(1, x_len);
    a_1 = zeros(interp_point_num, x_len);
    a_2 = zeros(interp_point_num, x_len);
    b_1 = zeros(interp_point_num, x_len);
    b_2 = zeros(interp_point_num, x_len);
    for i = 1:(interp_point_num-1),
        a_1(i,:) = (1 + 2.*((X - XX(i)) ./ (XX(i+1) - XX(i)))) .* ((X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        a_2(i,:) = (1 + 2.*((X - XX(i+1)) ./ (XX(i) - XX(i+1)))) .* ((X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
        b_1(i,:) = (X - XX(i)) .* ((X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        b_2(i,:) = (X - XX(i+1)) .* ((X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
        h_3 = h_3 + YY(i).*a_1(i,:) + YY(i+1).*a_2(i,:) + Yd(i).*b_1(i,:) + Yd(i+1).*b_2(i,:);
    end
end