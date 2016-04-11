function mN = my_wMAP( input, target, Phi, alpha, beta)
%MY_WMAP Summary of this function goes here
%   Detailed explanation goes here
    SN_inv = alpha * eye(size(input,2)+1) + beta * (Phi' * Phi);
    SN = inv(SN_inv);
    mN = beta * SN * Phi' * target; 
end

