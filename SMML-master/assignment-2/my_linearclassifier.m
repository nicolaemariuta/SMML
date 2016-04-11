function output = my_linearclassifier(x, covariance, mean, prior)
%MY_LINEAR Summary of this function goes here
%   Detailed explanation goes here

invCov = inv(covariance);
output = x * invCov * mean' - 1/2 * mean * invCov * mean' + log(prior);

end

