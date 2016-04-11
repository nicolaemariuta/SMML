function output = my_wML( Phi, T )
%MY_WML find maximum likelihood estimates
    output = pinv(Phi'*Phi) * (Phi'* T);
end

