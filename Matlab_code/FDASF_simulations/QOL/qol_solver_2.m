function [X_star,rho_star]=qol_solver_2(Ryy,B,C,d,Q)
% Function solving the RTLS problem, i.e.,
% min trace(X'*Ryy*X)+trace(X'*A) / (trace(X'*B)+c) s.t. trace(X'*B)+c > 0
    
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'Display','off','OptimalityTolerance',1e-8);

    fun = @(x)qolobjfunc(x,Ryy,B,C,d,Q);
    nonlconstr = @(x)linconstr(x,C,d,Q);
    
    M=size(Ryy,1);
    
    X=randn(M*Q,1);

    [X,fval,eflag,output,lambda] = fmincon(fun,X,...
        [],[],[],[],[],[],nonlconstr,options);
    
    X=reshape(X,[],Q);
        
    rho=qol_obj(X,Ryy,B,C,d);
    X_star=X;
    rho_star=rho;
end

function [y,grady] = qolobjfunc(x,Ryy,B,C,d,Q)
    X_mat=reshape(x,[],Q);
    y = (trace(X_mat'*Ryy*X_mat)+trace(X_mat'*B))/(trace(X_mat'*C)+d);
    if nargout > 1
        grady = 2*reshape(Ryy*X_mat,[],1)/trace(trace(X_mat'*C)+d)...
            +reshape(B,[],1)/trace(trace(X_mat'*C)+d)...
            -reshape(C,[],1)*(trace(X_mat'*Ryy*X_mat)+trace(X_mat'*B))/((trace(X_mat'*C)+d)^2);
            
    end
end

function [y,yeq,grady,gradyeq] = linconstr(x,C,d,Q)
    X_mat=reshape(x,[],Q);
    y = -trace(X_mat'*C)-d;
    yeq = [];

    if nargout > 2
        grady = reshape(C,[],1);
        gradyeq = [];
    end
end

