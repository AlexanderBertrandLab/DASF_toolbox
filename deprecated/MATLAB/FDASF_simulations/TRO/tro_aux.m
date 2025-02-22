function X=tro_aux(Ryy,Rvv,C,Q,rho)
% Function solving the auxiliary problem of the RTLS problem, i.e.,
% max trace(X'*Ryy*X)-rho*trace(X'*Rvv*X) s.t. X'*C*X = I_Q 

    [U_c,S_c,V_c]=svd(C);

    Kyy=sqrt(inv(S_c))*U_c'*Ryy*U_c*sqrt(inv(S_c));
    Kvv=sqrt(inv(S_c))*U_c'*Rvv*U_c*sqrt(inv(S_c));
    Kyy=make_sym(Kyy);
    Kvv=make_sym(Kvv);
    
    [eigvec_rho,eigval_rho]=eig(Kyy-rho*Kvv);
    [~,ind_int]=sort(diag(eigval_rho),'descend');
    X=eigvec_rho(:,ind_int(1:Q));
    
    X=U_c*sqrt(inv(S_c))*X;

end