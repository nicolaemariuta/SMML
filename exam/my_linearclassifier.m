function output = my_linearclassifier(x, covariance, mean, prior)
%calculate output of classifier for test data x based on the model that is
%built

invCov = inv(covariance);
output = x * invCov * mean' - 1/2 * mean * invCov * mean' + log(prior);

end

