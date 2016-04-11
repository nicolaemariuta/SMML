function error = my_ms( predicted, measured)
%calculate mean-squared error between predicted data and calculated value
    error = (1/length(predicted) * sum((measured - predicted).^2));
end

