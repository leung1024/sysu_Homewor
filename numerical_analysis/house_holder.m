function [nu, beta] = house_holder(x)
    n = length(x);
    eta = max(x);
    x = x / eta;
    sigma = x(2:n)' * x(2:n);
    nu(2:n) = x(2:n);
    if sigma == 0
        beta = 0;
    else
        alpha = sqrt(x(1)^2 + sigma);
        if x(1) <= 0
            nu(1) = x(1) - alpha;
        else
            nu(1) = -sigma/(x(1) + alpha);
        end
        beta = 2 * nu(1)^2 / (sigma + nu(1)^2);
        nu = nu / nu(1);
    end
end