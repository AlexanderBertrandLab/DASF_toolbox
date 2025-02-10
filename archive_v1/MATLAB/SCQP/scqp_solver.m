function [X_star,f_star]=scqp_solver(prob_params,data)
% 
% Solve the SCQP: min_X f(X)=0.5*E[||X'*y(t)||^2]+trace(X'*B) 
%                 s.t. trace(X'*Gamma*X)=1.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    B=data.B_cell{1};
    Gamma=data.Gamma_cell{1};
    nbsamples=prob_params.nbsamples;
    
    manifold = spherefactory(size(B,1),size(B,2));
    
    Ryy=make_sym(Y*Y')/nbsamples;
    
    L=chol(make_sym(Gamma))';
    Ryyt=inv(L)*Ryy*inv(L)';
    Ryyt=make_sym(Ryyt);
    Bt=inv(L)*B;
    
    problem.M = manifold;
    problem.cost  = @(X) 0.5*trace(X'*(Ryyt*X))+trace(X'*Bt);
    problem.egrad = @(X) Ryyt*X+Bt;
    warning('off', 'manopt:getHessian:approx') 
    options.verbosity=0;
    [X_star, f_star, info, options] = trustregions(problem,[],options);
    X_star=inv(L')*X_star;
        
        
end