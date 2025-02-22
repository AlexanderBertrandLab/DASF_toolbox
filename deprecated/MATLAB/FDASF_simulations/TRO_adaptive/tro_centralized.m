function [X,rho]=tro_centralized(Ryy,Rvv,C,Q)
% Function solving TRO problem, i.e.,
% max trace(X'*Ryy*X) / trace(X'*Rvv*X) s.t. X'*C*X = I_Q

    M=size(Ryy,1);
    i=0;
    xinit=randn(M,Q);
    rho=tro_obj(xinit,Ryy,Rvv);
    rho_old=rho+1;
    tol_rho=1e-12;
    nbiter=300;
    X=xinit;
    X_old=X+ones(size(X));
    [U_c,S_c,V_c]=svd(C);

    Kyy=sqrt(inv(S_c))*U_c'*Ryy*U_c*sqrt(inv(S_c));
    Kvv=sqrt(inv(S_c))*U_c'*Rvv*U_c*sqrt(inv(S_c));
    Kyy=make_sym(Kyy);
    Kvv=make_sym(Kvv);
    
    while (abs(rho-rho_old)>tol_rho) && (i<nbiter)
        X_old=X;
        [eigvec_rho,eigval_rho]=eig(Kyy-rho*Kvv);
        [~,ind_int]=sort(diag(eigval_rho),'descend');
        X=eigvec_rho(:,ind_int(1:Q));
        rho=tro_obj(X,Kyy,Kvv);

        i=i+1;
    end
    
    X=U_c*sqrt(inv(S_c))*X;
    rho=tro_obj(X,Ryy,Rvv);

end



