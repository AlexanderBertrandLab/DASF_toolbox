function f=tro_eval(X,data)

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