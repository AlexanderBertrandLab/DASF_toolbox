function X=qol_aux_solver(Ryy,B,C,rho)
% Function solving the auxiliary problem of the RTLS problem, i.e.,
% min trace(X'*Ryy*X)+trace(X'*A)-rho*(trace(X'*B)+c) s.t. trace(X'*B)+c > 0
    X=0.5*inv(Ryy)*(rho*C-B);
end