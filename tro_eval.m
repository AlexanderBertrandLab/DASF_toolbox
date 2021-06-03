function f=tro_eval(X,data)

% Evaluate the TRO objective trace(X'*Ryy*X)/trace(X'*Rvv*X);

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y_cell=data.Y_cell;

    Y=Y_cell{1};
    V=Y_cell{2};

    N=size(Y,2);

    Ryy=1/N*conj(Y*Y');
    Rvv=1/N*conj(V*V');
    Ryy=make_sym(Ryy);
    Rvv=make_sym(Rvv);

    f=trace(X'*Ryy*X)/trace(X'*Rvv*X);

end