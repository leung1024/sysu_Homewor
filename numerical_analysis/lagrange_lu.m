function L_x = lagrange_lu(X, XX, YY, N)
    % X     x轴刻度值
    % XX    插值点
    % YY    插值点的函数值
    % N   	Lagrange的阶数
    l_x = ones(N + 1, length(X));
    L_x = zeros(1, length(X));
    for i = 1:N+1,
        for j = 1:length(XX),
            if j ~= i,
                l_x(i,:) = l_x(i,:) .* ((X - XX(j)) ./ (XX(i) - XX(j)));
            end
        end
        L_x = L_x + (YY(i) * l_x(i,:));
    end
end
