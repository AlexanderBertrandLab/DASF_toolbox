function rho=qol_obj(X,Ryy,B,C,d)
    rho=(trace(X'*Ryy*X)+trace(X'*B))/(trace(X'*C)+d);
end
