function [X_star,rho_star]=rtls_solver_2(Ryy,ryd,rdd,LL,C,delta)
% Function solving the RTLS problem, i.e.,
% min (x'*Ryy*x-2*x'*ryd+rdd) / (1+x'*C*x) s.t. x'*LL*x <= delta^2 

    X=randn(size(Ryy,2),1);
    
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'Display','off');

    fun = @(x)tlsobjfunc(x,Ryy,ryd,rdd,C);
    nonlconstr = @(x)quadconstr(x,LL,delta);

    [X,fval,eflag,output,lambda] = fmincon(fun,X,...
        [],[],[],[],[],[],nonlconstr,options);
        
    rho=(X'*Ryy*X-2*X'*ryd+rdd)/(X'*C*X+1);
    X_star=X;
    rho_star=rho;
end

function [y,grady] = tlsobjfunc(x,Ryy,ryd,rdd,C)
    y = (x'*Ryy*x-2*x'*ryd+rdd)/(x'*C*x+1);
    if nargout > 1
        grady = 2*(Ryy*x-ryd)/(x'*C*x+1)-2*C*x*(x'*Ryy*x-2*x'*ryd+rdd)/(x'*C*x+1).^2;
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