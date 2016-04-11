function O = activationFunctionDerivate(I)
O = 1./(1+abs(I)).^2;
end

