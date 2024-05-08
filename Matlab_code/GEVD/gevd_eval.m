function f=gevd_eval(X,data)

% Evaluate the GEVD objective E[||X'*y(t)||^2].

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y_cell=data.Y_cell;

    Y=Y_cell{1};
    V=Y_cell{2};

    N=size(Y,2);

    Ryy=make_sym(Y*Y')/N;
    Rvv=make_sym(V*V')/N;

    f=trace(X'*Ryy*X);

end