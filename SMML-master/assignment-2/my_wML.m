function output = my_wML( Phi, T )
%MY_WML Summary of this function goes here
%find maximum likelihood estimates
    output = pinv(Phi'*Phi) * (Phi'* T);
end

