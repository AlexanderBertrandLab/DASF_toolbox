function [X_star,f_star]=qcqp_solver(data,prob_params)

% Solve min 0.5*trace(X'*Ryy*X)-trace(B'*X) s.t. trace(X'*Gamma*X)<= alpha^2; X'*c=d.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Y=data.Y_cell{1};
    B=data.B_cell{1};
    c=data.B_cell{2};
    Gamma=data.Gamma_cell{1};
    alpha=data.Glob_Const_cell{1};
    d=data.Glob_Const_cell{2};
    nbsamples=prob_params.nbsamples;
    
    Ryy=1/nbsamples*conj(Y*Y');
    Ryy=make_sym(Ryy);

    [U,S,V]=svd(Gamma);
    sqrt_S=diag(sqrt(diag(S)));
    sqrt_Gamma=(U*sqrt_S)';

    if (alpha^2==norm(d)^2/norm(inv(sqrt_Gamma)'*c)^2)

        X_star=inv(Gamma)*c*d'/norm(sqrt_Gamma'*c);

    elseif (alpha^2>norm(d)^2/norm(inv(sqrt_Gamma)'*c)^2)

        Mf=@(muv)Ryy+muv*Gamma;
        wf=@(muv)(B'*inv(Mf(muv))'*c-d)/(c'*inv(Mf(muv))*c);
        Xf=@(muv)inv(Mf(muv))*(B-c*wf(muv)');
        norm_fun=@(muv)trace(Xf(muv)'*Gamma*Xf(muv))-alpha^2;
        f_fun=@(muv)0.5*trace(Xf(muv)'*Ryy*Xf(muv))-trace(B'*Xf(muv));
        if norm_fun(0)<0
            X_star=Xf(0);
        else 
            mu_star=fzero(norm_fun,[0,1e15]);
            X_star=Xf(mu_star);
        end

    else
        warning('Infeasible')
    end
        f_star=qcqp_eval(X_star,data);
        
end

