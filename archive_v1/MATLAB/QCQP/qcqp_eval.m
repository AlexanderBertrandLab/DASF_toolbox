function f=qcqp_eval(X,data)

% Evaluate the QCQP objective 0.5*E[||X'*y(t)||^2]-trace(X'*B);

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    B=data.B_cell{1};
    
    N=size(Y,2);
    Ryy=make_sym(Y*Y')/N;
    
    f=0.5*trace(X'*Ryy*X)-trace(X'*B);
    
end