function f = hermite_lu(X, XX, YY, Yd)
    f = zeros(1, length(X));
    n = length(XX);
    for i=1:n
        h = 1.0;
        a = 0.0;
        for j=1:n
            if( j ~= i)
                h = h.*(X-XX(j)).^2./((XX(i)-XX(j)).^2);
                a = a + 1/(XX(i)-XX(j));
            end
        end
        f= f + h.*((XX(i)-X).*(2.*a.*YY(i)-Yd(i))+YY(i));
    end

end