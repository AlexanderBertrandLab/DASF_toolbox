function f=ls_eval(X,data)

% Evaluate the LS objective E[||d(t)-X'*y(t)||^2];

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    D=data.Glob_Const_cell{1};
    
    N=size(Y,2);
    Ryy=make_sym(Y*Y')/N;
    Rdd=make_sym(D*D')/N;
    Ryd=Y*D'/N;
    
    f=trace(X'*Ryy*X)+trace(Rdd)-2*trace(X'*Ryd);
    
end