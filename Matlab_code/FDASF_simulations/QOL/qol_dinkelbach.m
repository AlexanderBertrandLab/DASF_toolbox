function [X,iter_count,rho]=qol_dinkelbach(Ryy,B,C,d)
% Function solving the RTLS problem, i.e.,
% min trace(X'*Ryy*X)+trace(X'*A) / (trace(X'*B)+c) s.t. trace(X'*B)+c > 0

    M=size(Ryy,1);
    Q=size(B,2);
    i=0;
    xinit=randn(M,Q);
    rho=qol_obj(xinit,Ryy,B,C,d);
    nbiter=10;
    X=xinit;
    X_old=X+ones(size(X));
    tol_X=1e-8;
    
    while (norm(X-X_old,'fro')>tol_X) && (i<nbiter)
        X_old=X;
        X=qol_aux_solver(Ryy,B,C,rho);
        rho=qol_obj(X,Ryy,B,C,d);
        i=i+1;
    end
    
    iter_count=i;

end

