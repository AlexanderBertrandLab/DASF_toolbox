function [X,f]=tro_solver(prob_params,data)

    i=0;
    M=size(data.Y_cell{1},1);
    Q=prob_params.Q;
    Xinit=randn(M,Q);
    Xinit=normc(Xinit);
    f=tro_eval(Xinit,data);
    f_old=f+1;
    tol_f=1e-10;
    nbiter=-1;
    X=Xinit;

    Y=data.Y_cell{1};
    V=data.Y_cell{2};
    Gamma=data.Gamma_cell{1};

    N=size(Y,2);

    Ryy=1/N*conj(Y*Y');
    Rvv=1/N*conj(V*V');
    Ryy=make_sym(Ryy);
    Rvv=make_sym(Rvv);

    [U_c,S_c,V_c]=svd(Gamma);

    Y_t=sqrt(inv(S_c))*U_c'*Y;
    V_t=sqrt(inv(S_c))*U_c'*V;

    Kyy=sqrt(inv(S_c))*U_c'*Ryy*U_c*sqrt(inv(S_c));
    Kvv=sqrt(inv(S_c))*U_c'*Rvv*U_c*sqrt(inv(S_c));
    Kyy=make_sym(Kyy);
    Kvv=make_sym(Kvv);

    while (tol_f>0 && abs(f-f_old)>tol_f) || (i<nbiter)

        [E_int,l_int]=eig(Kyy-f*Kvv);
        [~,ind_int]=sort(diag(l_int),'descend');

        X=E_int(:,ind_int(1:Q));
        f_old=f;
        data_t=struct;
        Y_cell_t=cell(2,1);
        Y_cell_t{1}=Y_t;
        Y_cell_t{2}=V_t;
        data_t.Y_cell=Y_cell_t;
        f=tro_eval(X,data_t);

        i=i+1;

    end

    X=U_c*sqrt(inv(S_c))*X;

end

