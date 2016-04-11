function output = my_normalize( data, means, stds )
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here
    output = data(:,1:2) - repmat(means, size(data,1), 1);
    output = output ./ repmat(stds, size(data,1), 1);
    output = [ output data(:,3) ];
end