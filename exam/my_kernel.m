function k = my_kernel(x, y, gamma)
    I = size(x,1);
    J = size(y,1);
    k = zeros(I,J);
    for i=1:I
        for j=1:J
            q = x(i,:) - y(j,:);
            k(i,j) = exp(- gamma * q * q');
        end
    end
end

