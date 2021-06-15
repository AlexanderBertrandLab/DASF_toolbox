function [X,f_seq,norm_diff,norm_err]=dsfo(prob_params,data,conv,...
    obj_eval,prob_solver,prob_resolve_uniqueness)

% Function running the TI-DSFO for a given problem.
%
% INPUTS :
% prob_params : Structure related to the problem parameters containing the
%               following fields.
%            nbnodes : Number of nodes in the network.
%            nbsensors_vec : Vector containing the number of sensors for
%                            each node.
%            nbsensors : Sum of the number of sensors for each node
%                        (dimension of the network-wide signals). Is equal
%                        to sum(nbsensnode).
%            Q : Number of filters to use (dimension of projected space)
%            nbsamples : Number of time samples of the signals per
%                        iteration.
%            graph_adj : Adjacency (binary) matrix, with graph_adj(i,j)=1  
%                        if i and j are connected. Otherwise 0. 
%                        graph_adj(i,i)=0.
%            X_star: (Optional) Optimal argument solving the the problem
%                    (for comparison, e.g., to compute norm_err).
%            compare_opt: If equal to 1 and X_star is given, compute
%                         norm_err.
%            plot_dynamic: If equal to 1 and X_star is given, plot
%                          dynamically the first column of X_star and the 
%                          current estimate X.
%
% data        : Structure related data containing the following fields:
%            Y_cell : Cell containing matrices of size 
%                     (nbsensors x nbsamples) corresponding to the
%                     stochastic signals.
%            B_cell : Cell containing matrices or vectors with (nbsamples)
%                     rows corresponding to the constant parameters.
%            Gamma_cell : Cell containing matrices of size 
%                         (nbsensors x nbsensors) corresponding to the
%                         quadratic parameters.
%            Glob_Const_cell : Cell containing the global constants which
%                         are not filtered through X.
%
% conv        : Structure related to the convergence and stopping criteria
%               of the algorithm, containing the following fields
%            tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
%            nbiter : Max. nb. of iterations.
% If both values are valid, the algorithm will continue until the stricter
% condition (OR). One of the criteria can be chosen explicitly by
% initializing the other to a negative value.
%
% obj_eval    : Function evaluating the objective of the problem.
%
% prob_solver : Function solving the centralized problem.
%
% prob_resolve_uniqueness : (Optional) Function resolving the uniqueness
%                           ambiguity.
%
%
% OUTPUTS : 
% X               : Estimation of the optimal variable
% f_seq           : Sequence of objective values across iterations
% norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q)
% norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be


Q=prob_params.Q;
nbsensors=prob_params.nbsensors;
nbnodes=prob_params.nbnodes;
nbsensors_vec=prob_params.nbsensors_vec;
adj=prob_params.graph_adj;
if (~isfield(prob_params,'X_star'))
    X_star=[];
else
    X_star=prob_params.X_star;
end
if (~isfield(prob_params,'compare_opt'))
    compare_opt=0;
else
    compare_opt=prob_params.compare_opt;
end
if (~isfield(prob_params,'plot_dynamic'))
    plot_dynamic=0;
else
    plot_dynamic=prob_params.plot_dynamic;
end

tol_f=conv.tol_f;
nbiter=conv.nbiter;

X=randn(nbsensors,Q);
X=normc(X);
f=obj_eval(X,data);

i=0;
f_old=f+1;

f_seq=[];
norm_diff=[];
norm_err=[];

path=1:nbnodes;
% Random updating order
rand_path=path(randperm(length(path)));

while (tol_f>0 && abs(f-f_old)>tol_f) || (i<nbiter)
    
    % Select updating node.
    q=rand_path(rem(i,nbnodes)+1);
    
    % Prune the network.
    % Find shortest path.
    [neighbors,path]=find_path(q,adj);
    
    % Neighborhood clusters.
    Nu=constr_Nu(neighbors,path);
    
    % Global - local transition matrix.
    Cq=constr_Cq(X,q,prob_params,neighbors,Nu);

    % Compute compressed data.
    data_compressed=compress(data,Cq);

    % Compute the local variable.
    % Solve the local problem using the algorithm for the global problem
    % using compressed data.
    [X_tilde,~]=prob_solver(prob_params,data_compressed);
    
    % Evaluate objective.
    f_old=f;
    f=obj_eval(X_tilde,data_compressed);
    f_seq=[f_seq,f];
    
    % Global variable.
    X=Cq*X_tilde;
    
    if i>0
        if(~isempty(prob_resolve_uniqueness))
            Xq=block_q(X,q,nbsensors_vec);
            Xq_old=block_q(X_old,q,nbsensors_vec);
            X=prob_resolve_uniqueness(Xq_old,Xq,X);
        end
        norm_diff=[norm_diff,norm(X-X_old,'fro').^2/numel(X)];
    end
    
    if(~isempty(X_star) && compare_opt==1)
        if(~isempty(prob_resolve_uniqueness))
            Xq=block_q(X,q,nbsensors_vec);
            Xq_star=block_q(X_star,q,nbsensors_vec);
            X=prob_resolve_uniqueness(Xq_star,Xq,X);
        end
        norm_err=[norm_err,norm(X-X_star,'fro')^2/norm(X_star,'fro')^2];
        if(plot_dynamic==1)
            dynamic_plot(X,X_star)
        end
    end
    
    X_old=X;
    
    i=i+1;

end

end

function dynamic_plot(X,X_star)

    plot(X_star(:,1),'r')
    hold on
    plot(X(:,1),'b')
    ylim([1.2*min(real(X_star(:,1))) 1.2*max(real(X_star(:,1)))]);
    hold off
    drawnow
    
end


