function [X_star,f_star]=cca_solver(prob_params,data)

% Solve the CCA Problem: max_(X,W) E[trace(X'*y(t)*v(t)'*W)] 
%                        s.t. E[X'*y(t)*y(t)'*X]=I, E[W'*v(t)*v(t)'*W]=I.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    data_X=data{1};
    data_W=data{2};
    Y=data_X.Y_cell{1};
    V=data_W.Y_cell{1};
    nbsamples=prob_params.nbsamples;
    Q=prob_params.Q;
    
    Ryy=make_sym(Y*Y')/nbsamples;
    Rvv=make_sym(V*V')/nbsamples;
    Ryv=Y*V'/nbsamples;
    Rvy=Ryv';
    
    inv_Rvv=make_sym(inv(Rvv));
    A_X=make_sym(Ryv*inv_Rvv*Rvy);
    
    X_star=cell(2,1);
    [E_X,L_X]=eig(A_X,Ryy);
    l_X=diag(L_X);
    [~,ind]=sort(l_X,'descend');
    l_X=l_X(ind);
    E_X=E_X(:,ind);
    X_star{1}=E_X(:,1:Q);
    E_W=inv_Rvv*Rvy*E_X*diag(1./sqrt(abs(l_X)));
    X_star{2}=E_W(:,1:Q);
    
    f_star=trace(X_star{1}'*Ryv*X_star{2});

end

