function [X,norm_diff,norm_err,f_seq]=dasf_multivar(prob_params,data,...
    prob_solver,conv,prob_select_sol,prob_eval)

% Function running the DASF for a given problem.
%
% INPUTS :
% prob_params : Structure related to the problem parameters containing the
%               following fields:
%            nbnodes : Number of nodes in the network.
%            nbsensors_vec : Vector containing the number of sensors for
%                            each node.
%            nbsensors : Sum of the number of sensors for each node
%                        (dimension of the network-wide signals). Is equal
%                        to sum(nbsensors_vec).
%            Q : Number of filters to use (dimension of projected space)
%            nbsamples : Number of time samples of the signals per
%                        iteration.
%            nbvariables : Number of variables.
%            graph_adj : Adjacency (binary) matrix, with graph_adj(i,j)=1  
%                        if i and j are connected. Otherwise 0. 
%                        graph_adj(i,i)=0.
%            update_path : (Optional) Vector of nodes representing the 
%                          updating path followed by the algorithm. If not 
%                          provided, a random path is created.
%            X_init : (Optional) Initial estimate for X.
%            X_star : (Optional) Optimal argument solving the problem
%                     (for comparison, e.g., to compute norm_err).
%            compare_opt : (Optional, binary) If "true" and X_star is given, 
%                          compute norm_err. "false" by default.
%            plot_dynamic : (Optional, binary) If "true" X_star is given, 
%                           plot dynamically the first column of X_star and 
%                           the current estimate X. "false" by default.
%
% data : Structure related to the data containing the following fields:
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
% prob_solver : Function solving the centralized problem.
%
% conv:  (Optional) Structure related to the convergence and stopping 
%         criteria of the algorithm, containing the following fields:
%
%            nbiter : Maximum number of iterations.
%
%            tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
%
%            tol_X : Tolerance in arguments: ||X^(i+1)-X^(i)||_F>tol_f
%
% By default, the number of iterations is 200, unless specified otherwise.
% If other fields are given and valid, the first condition to be achieved
% stops the algorithm.
%
% prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.
%
% prob_eval : (Optional) Function evaluating the objective of the problem.
%                           
%
%
% OUTPUTS : 
% X               : Estimation of the optimal variable.
% norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q).
% norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2.
% f_seq           : Sequence of objective values across iterations.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be


Q=prob_params.Q;
nbsensors=prob_params.nbsensors;
nbnodes=prob_params.nbnodes;
nbsensors_vec=prob_params.nbsensors_vec;
nbvariables=prob_params.nbvariables;
graph_adj=prob_params.graph_adj;
rng('shuffle');

if (~isfield(prob_params,'update_path'))
    % Random updating order.
    update_path=randperm(nbnodes);
    warning('Randomly selected updating path')
else
    update_path=prob_params.update_path;
end

compare_opt_flag=false;
plot_dynamic_flag=false;

if (~isfield(prob_params,'X_star'))
    X_star=[];
    compare_opt_flag=true;
    plot_dynamic_flag=true;
else
    X_star=prob_params.X_star;
end

if (~isfield(prob_params,'compare_opt') || compare_opt_flag)
    compare_opt=false;
else
    compare_opt=prob_params.compare_opt;
end

if (~isfield(prob_params,'plot_dynamic') || plot_dynamic_flag)
    plot_dynamic=false;
else
    plot_dynamic=prob_params.plot_dynamic;
end

if (isempty(conv))
    tol_f_break=false;
    tol_X_break=false;
    nbiter=200;
    warning('Performing 200 iterations')
elseif ((isfield(conv,'nbiter') && conv.nbiter>0) || ...
                            (isfield(conv,'tol_f') && conv.tol_f>0) || ...
                            (isfield(conv,'tol_X') && conv.tol_X>0))
    
    if (~isfield(conv,'nbiter') || conv.nbiter<=0)
        nbiter=200;
        warning('Performing at most 200 iterations')
    else
        nbiter=conv.nbiter;
    end
                        
    if (~isfield(conv,'tol_f') || conv.tol_f<=0)
        tol_f_break=false;
    else
        tol_f=conv.tol_f;
        tol_f_break=true;
    end
    if (~isfield(conv,'tol_X') || conv.tol_X<=0)
        tol_X_break=false;
    else
        tol_X=conv.tol_X;
        tol_X_break=true;
    end
else
    tol_f_break=false;
    tol_X_break=false;
    nbiter=200;
    warning('Performing 200 iterations')
end

X=cell(nbvariables,1);

if (~isfield(prob_params,'X_init'))
    for k=1:nbvariables
        X{k}=randn(nbsensors,Q);
    end
else
    X=prob_params.X_init;
end

X_old=X;

if(isempty(prob_eval))
    tol_f_break=false;
else
    f=prob_eval(X,data);
end

i=0;

f_seq=[];
norm_diff=[];

X_cell_track={};

while i<nbiter
    
    % Select updating node.
    q=update_path(rem(i,nbnodes)+1);
    
    % Prune the network.
    % Find shortest path.
    [neighbors,path]=find_path(q,graph_adj);
    
    % Neighborhood clusters.
    clusters=find_clusters(neighbors,path);
    
    % Global - local transition matrices.
    Cq=cell(nbvariables,1);
    for k=1:nbvariables
       Cq{k}=build_Cq(X{k},q,prob_params,neighbors,clusters); 
    end

    % Compute the compressed data.
    data_compressed=cell(nbvariables,1);
    for k=1:nbvariables
        data_compressed{k}=compress(data{k},Cq{k});
    end
    
    % Compute the local variable.
    % Solve the local problem with the algorithm for the global problem
    % using the compressed data.
    X_tilde=prob_solver(prob_params,data_compressed);
        
    % Select a solution among potential ones if the problem has a non-
    % unique solution.
    if(~isempty(prob_select_sol))
        X_tilde_old=cell(nbvariables,1);
        for k=1:nbvariables
            Xq_old=block_q(X_old{k},q,nbsensors_vec);
            X_tilde_old{k}=[Xq_old;repmat(eye(Q),length(neighbors),1)];
        end
        X_tilde=prob_select_sol(X_tilde_old,X_tilde,prob_params,data_compressed,q);
    end
    
    % Evaluate the objective.
    if(~isempty(prob_eval))
        f_old=f;
        f=prob_eval(X_tilde,data_compressed);
        f_seq=[f_seq,f];
    end
    
    % Global variable.
    for k=1:nbvariables
        X{k}=Cq{k}*X_tilde{k};
    end
    
    if i>0
        norm_diff=[norm_diff,norm(cell2mat(X)-cell2mat(X_old),'fro').^2/...
            numel(cell2mat(X))];
    end
    
    if(plot_dynamic)
        if(~isempty(prob_select_sol))
            X_compare=prob_select_sol(X_star,X,prob_params,data,q);
        end
        dynamic_plot(cell2mat(X_compare),cell2mat(X_star))
    end
    
    X_old=X;
    
    i=i+1;
    
    for k=1:nbvariables
        X_cell_track{i,k}=X{k};
    end
    
    if (tol_f_break && abs(f-f_old)<=tol_f) || (tol_X_break &&...
            norm(cell2mat(X)-cell2mat(X_old),'fro')<=tol_X)
        break
    end

end

if(compare_opt)
    % Resolve uniqueness ambiguity on X_star for comparison
    if(~isempty(prob_select_sol))
        X_star=prob_select_sol(X,X_star,prob_params,data,q);
    end
    
    total_iterations=length(X_cell_track);
    norm_err=zeros(1,total_iterations);
    for i=1:total_iterations
        X_cell_track_i=cell(nbvariables,1);
        for k=1:nbvariables
            X_cell_track_i{k}=X_cell_track{i,k};
        end
        norm_err(i)=norm(cell2mat(X_cell_track_i)-cell2mat(X_star),'fro')^2/...
            norm(cell2mat(X_star),'fro')^2;
    end
end

end




