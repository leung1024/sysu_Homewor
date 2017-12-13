function f = thrid_order_hermite_lu(X, XX, YY, Yd)
    x_len = length(X);
    interp_point_num = length(XX)
    f = zeros(1, x_len);
    a_1 = zeros(x_len)
    for i = 1:(interp_point_num-1),
        a_1(i) = (1 .+ 2.*((X - XX(i)) ./ (XX(i+1) - XX(i)))) .* ((X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        a_2(i) = (1 .+ 2.*((X - XX(i+1)) ./ (XX(i+1) - XX(i)))) .* ((X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
        b_1(i) = (X - XX(i)) .* ((X - XX(i+1)) ./ (XX(i) - XX(i+1))).^2;
        b_2(i) = (X - XX(i+1)) .* ((X - XX(i)) ./ (XX(i+1) - XX(i))).^2;
    for i=1:interp_point_num
        h = 1.0;
        a = 0.0;
        for j=1:interp_point_num
            if( j ~= i)
                h = h.*(X-XX(j)).^2./((XX(i)-XX(j)).^2);
                a = a + 1/(XX(i)-XX(j));
            end
        end
        f= f + h.*((XX(i)-X).*(2.*a.*YY(i)-Yd(i))+YY(i));
    end

end