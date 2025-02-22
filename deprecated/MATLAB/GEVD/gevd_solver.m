function [X_star,f_star]=gevd_solver(prob_params,data)

% Solve the GEVD Problem: max_X E[||X'*y(t)||^2]
%                         s.t. E[X'*v(t)*v(t)'*X]=I.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Q=prob_params.Q;

    Y=data.Y_cell{1};
    V=data.Y_cell{2};

    N=prob_params.nbsamples;

    Ryy=make_sym(Y*Y')/N;
    Rvv=make_sym(V*V')/N;
    
    [eigvecs,eigvals]=eig(Ryy,Rvv);
    [~,ind]=sort(diag(eigvals),'descend');

    X_star=eigvecs(:,ind(1:Q));
    f_star=gevd_eval(X_star,data);

end

