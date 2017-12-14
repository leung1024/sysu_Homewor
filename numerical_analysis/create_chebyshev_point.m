function t = create_chebyshev_point(N)
    t = zeros(1, N);
    for i = 1:N,
        t(i) = cos((2*i - 1) * pi / (2*N));
    end
end

% x = linspace(-10, 10, 8000);
% y = x.*exp(x);
% t = create_chebyshev_point(11);
% x_c = 5*t;
% y_c = x_c.*exp(x_c);
% f = lagrange_lu(x, x_c, y_c, 10);
% figure()
% plot(x,y)
% hold on
% plot(x,f)

% x = linspace(-5, 5, 8000);
% y = 1./(1+x.^2);
% t = create_chebyshev_point(11);
% x_c = 5*t;
% y_c = 1./(1+x_c.^2);
% f = lagrange_lu(x, x_c, y_c, 10);
% figure()
% plot(x,y)
% hold on
% plot(x,f)