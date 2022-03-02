function [X_star,f_star]=lcmv_solver(prob_params,data)

% Solve the LCMV Problem: min_X E[||X'*y(t)||^2] 
%                         s.t. X'*B=H.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    B=data.B_cell{1};
    H=data.Glob_Const_cell{1};
    nbsamples=prob_params.nbsamples;
    
    Ryy=make_sym(Y*Y')/nbsamples;

    X_star=inv(Ryy)*B*inv(B.'*inv(Ryy)*B)*H;
    f_star=lcmv_eval(X_star,data);
        
end

