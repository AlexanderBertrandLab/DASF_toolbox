function rho=tro_obj(X,Ryy,Rvv)
    rho=trace(X'*Ryy*X)/trace(X'*Rvv*X);
end
