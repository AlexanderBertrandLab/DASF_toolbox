function f=cca_eval(X_cell,data)

% Evaluate the CCA objective E[trace(X'*y(t)*v(t)'*W)];

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    data_X=data{1};
    data_W=data{2};
    Y=data_X.Y_cell{1};
    V=data_W.Y_cell{1};
    
    N=size(Y,2);
    Ryv=Y*V'/N;
    X=X_cell{1};
    W=X_cell{2};
    
    f=trace(X'*Ryv*W);

end