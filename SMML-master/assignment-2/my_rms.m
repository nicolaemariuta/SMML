function error = my_rms( predicted, measured)
%MY_RMS Summary of this function goes here
%   Detailed explanation goes here
    error = sqrt(1/length(predicted) * sum((measured - predicted).^2));
end

