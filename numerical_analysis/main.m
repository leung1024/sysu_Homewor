clc
clear
x = linspace(-5, 5, 101);
y = 1 ./ (1+x.^2);
% yd = (-2.*x) / (1 + x.^2).^2;

interp_num = 11;
xx = linspace(-5, 5, interp_num);
yy = 1 ./ (1+xx.^2);
yyd = (-2.*xx) ./ (1 + xx.^2).^2;

tic
L_x = lagrange_lu(x, xx, yy, interp_num-1);
figure(1)
plot(x, y)
hold on
plot(x, L_x)
L_r = sum(abs(y - L_x)) / 101;
toc

tic
H_x = hermite_lu(x, xx, yy, yyd);
figure(2)
plot(x, y)
hold on
plot(x, H_x)
H_r = sum(abs(y - H_x)) / 101;
toc

tic
t = create_chebyshev_point(11);
x_c = 5*t;
y_c = 1./(1+x_c.^2);
Lc_x = lagrange_lu(x, x_c, y_c, 10);
figure(3)
plot(x,y)
hold on
plot(x,Lc_x)
Lc_r = sum(abs(y - Lc_x)) / 101;
toc

tic
Li_x = liner_interp(x, xx, yy);
figure(4)
plot(x, y)
hold on
plot(x, Li_x)
Li_r = sum(abs(y - Li_x)) / 101;
toc

tic
H3_x = third_order_hermite_lu(x, xx, yy, yyd);
figure(5)
plot(x, y)
hold on
plot(x, H3_x)
H3_r = sum(abs(y - H3_x)) / 101;
toc


figure(6)
plot(x, y)
hold on
plot(x, L_x)
hold on
plot(x, H_x)
hold on
plot(x,Lc_x)
hold on
plot(x, Li_x)
hold on
plot(x, H3_x)
