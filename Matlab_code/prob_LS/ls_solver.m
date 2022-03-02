function [X_star,f_star]=ls_solver(prob_params,data)

% Solve the LS problem: min_X E[||d(t)-X'*y(t)||^2].

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    D=data.Glob_Const_cell{1};
    nbsamples=prob_params.nbsamples;
    
    Ryy=make_sym(Y*Y')/nbsamples;
    Ryd=Y*D'/nbsamples;

    X_star=inv(Ryy)*Ryd;
    f_star=ls_eval(X_star,data);
        
end

