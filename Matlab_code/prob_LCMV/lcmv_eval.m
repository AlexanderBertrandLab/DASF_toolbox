function f=lcmv_eval(X,data)

% Evaluate the LCMV objective E[||X'*y(t)||^2];

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    
    N=size(Y,2);
    Ryy=make_sym(Y*Y')/N;
    
    f=trace(X'*Ryy*X);
    
end