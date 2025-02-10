function rho=tls_obj(Ryy,ryd,rdd,x,C)

    rho=(x'*Ryy*x-2*x'*ryd+rdd)/(1+x'*C*x);
    
end