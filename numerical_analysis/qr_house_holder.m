function [A2, d] = qr_house_holder(A1)
    [m, n] = size(A1);
    A2 = zeros(m, n);
    for j = 1:n
        if j < m
            [nu, beta] = house_holder(A1(j:m,j))
            A2(j:m,j:n) = (eye(m-j+1) - beta * nu' * nu) * A1(j:m,j:n);
            d(j) = beta;
            A2(j+1:m,j) = nu(2:m-j+1);
        end
    end
end