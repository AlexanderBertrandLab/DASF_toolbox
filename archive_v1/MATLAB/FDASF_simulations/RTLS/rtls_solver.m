function [X_star,rho_star,nb_ind]=rtls_solver(Ryy,ryd,rdd,LL,C,delta)
% Function solving the RTLS problem using Dinkelbach's procedure, i.e.,
% min (x'*Ryy*x-2*x'*ryd+rdd) / (1+x'*C*x) s.t. x'*LL*x <= delta^2 

    X=randn(size(Ryy,2),1);
    rho=(X'*Ryy*X-2*X'*ryd+rdd)/(X'*C*X+1);
    nb_ind=0;
    X_old=X+ones(size(X));
    tol_X=1e-8;
    nbiter=10;
    i=0;
    while (norm(X-X_old,'fro')>tol_X) && (i<nbiter)
        
        X_old=X;
        
        options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
        'HessianFcn',@(x,lambda)quadhess(x,lambda,Ryy,rho,LL,C),'Display','off');

        fun = @(x)quadobj(x,Ryy,ryd,rdd,C,rho);
        nonlconstr = @(x)quadconstr(x,LL,delta);

        [X,fval,eflag,output,lambda] = fmincon(fun,X,...
            [],[],[],[],[],[],nonlconstr,options);
        
        rho=(X'*Ryy*X-2*X'*ryd+rdd)/(X'*C*X+1);
        nb_ind=nb_ind+1;
        i=i+1;
    end
    X_star=X;
    rho_star=rho;
end

function [y,grady] = quadobj(x,Ryy,ryd,rdd,C,rho)
    y = (x'*Ryy*x-2*x'*ryd+rdd)-rho*(x'*C*x+1);
    if nargout > 1
        grady = 2*(Ryy-rho*C)*x-2*ryd;
    end
end

function [y,yeq,grady,gradyeq] = quadconstr(x,LL,delta)
    y = x'*LL*x-delta^2;
    yeq = [];

    if nargout > 2
        grady = 2*LL*x;
        gradyeq = [];
    end
end

function hess = quadhess(x,lambda,Ryy,rho,LL,C)
    hess = 2*Ryy-2*rho*C+2*lambda.ineqnonlin*LL;
end