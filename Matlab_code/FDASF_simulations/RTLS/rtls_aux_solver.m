function x_star=rtls_aux_solver(Ryy,ryd,rdd,LL,C,delta,rho)
% Function solving the auxiliary problem of the RTLS problem, i.e.,
% min x'*Ryy*x-2*x'*ryd+rdd-rho*(1+x'*C*x) s.t. x'*LL*x <= delta^2 

    X=randn(size(Ryy,2),1);
    X=normc(X);
    
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'HessianFcn',@(x,lambda)quadhess(x,lambda,Ryy,rho,LL,C));

    fun = @(x)quadobj(x,Ryy,ryd,rdd,C,rho);
    nonlconstr = @(x)quadconstr(x,LL,delta);

    [X,fval,eflag,output,lambda] = fmincon(fun,X,...
        [],[],[],[],[],[],nonlconstr,options);

    rho=tls_obj(Ryy,ryd,rdd,X,C);
    
    x_star=X;
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
