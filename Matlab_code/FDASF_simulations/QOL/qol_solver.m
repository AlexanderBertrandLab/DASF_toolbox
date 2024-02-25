function [X_star,rho_star]=qol_solver(Ryy,B,C,d)
% Function solving the RTLS problem, i.e.,
% min trace(X'*Ryy*X)+trace(X'*A) / (trace(X'*B)+c) s.t. trace(X'*B)+c > 0 
    
    invR=inv(Ryy);
    invR=make_sym(invR);
    a=-trace(C'*invR*C)/4;
    b=d-trace(C'*invR*B)/2;
    c=-trace(B'*invR*B)/4;
    delta=b^2-4*a*c;
    lambda1=(-b+sqrt(delta))/(2*a);
    lambda2=(-b-sqrt(delta))/(2*a);
    X_star1=-0.5*invR*(B+lambda1*C);
    X_star2=-0.5*invR*(B+lambda2*C);
    rho_star1=qol_obj(X_star1,Ryy,B,C,d);
    rho_star2=qol_obj(X_star2,Ryy,B,C,d);
    
    if trace(X_star1'*C)+d>0
        X_star=X_star1;
        rho_star=rho_star1;
    else
        X_star=X_star2;
        rho_star=rho_star2;
    end
        
end