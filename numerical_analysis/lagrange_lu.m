function [L_x, l_x] = lagrange_lu(X, Y, XX, YY, N)
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
