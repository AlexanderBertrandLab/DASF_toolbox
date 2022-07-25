function [X,norm_diff,norm_err,f_seq]=dasf_async(prob_params,data,...
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
%            graph_adj : Adjacency (binary) matrix, with graph_adj(i,j)=1  
%                        if i and j are connected. Otherwise 0. 
%                        graph_adj(i,i)=0.
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
graph_adj=prob_params.graph_adj;
rng('shuffle');

if (~isfield(prob_params,'X_star'))
    X_star=[];
else
    X_star=prob_params.X_star;
end

if (~isfield(prob_params,'compare_opt'))
    compare_opt=false;
else
    compare_opt=prob_params.compare_opt;
end

if (~isfield(prob_params,'plot_dynamic'))
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

if (~isfield(prob_params,'X_init'))
    X=randn(nbsensors,Q);
else
    X=prob_params.X_init;
end

X_block=mat2cell(X,nbsensors_vec,Q);
X_old=X;

if(isempty(prob_eval))
    tol_f_break=false;
else
    f=prob_eval(X,data);
end

i=0;

f_seq=[];
norm_diff=[];
norm_err=[];

while i<nbiter
    
    alpha=1;
    
    nbupdating_nodes=randi(nbnodes);
    updating_nodes=randsample(nbnodes,nbupdating_nodes);
    %updating_nodes=[1:nbnodes];
    
    for l=1:length(updating_nodes)
        % Select updating node.
        q=updating_nodes(l);

        % Prune the network.
        % Find shortest path.
        [neighbors,path]=find_path(q,graph_adj);

        % Neighborhood clusters.
        clusters=find_clusters(neighbors,path);

        % Global - local transition matrix.
        Cq=build_Cq(X,q,prob_params,neighbors,clusters);

        % Compute the compressed data.
        data_compressed=compress(data,Cq);

        % Compute the local variable.
        % Solve the local problem with the algorithm for the global problem
        % using the compressed data.
        X_tilde=prob_solver(prob_params,data_compressed);
        nbneighbors=length(neighbors);
        X_tilde=prob_select_sol([X_block{q};repmat(eye(Q),nbneighbors,1)],...
            X_tilde,prob_params,data_compressed,q);
        
        Xq_new=X_tilde(1:nbsensors_vec(q),:);
        X_block{q}=(1-alpha)*X_block{q}+alpha*Xq_new;
        
    end
    
    
    %{
    CX=blkdiag(X_block{1:nbnodes});
    data_compressed=compress(data,CX);
    G=prob_solver(prob_params,data_compressed);
    X=CX*G;
    X_block=mat2cell(X,nbsensors_vec,Q);
    X_block=update_X_block(X_block,X_tilde,q,prob_params,neighbors,...
    clusters,prob_select_sol);
    X=cell2mat(X_block);
    %}
    
    
    X=cell2mat(X_block);
    q=1;
    [neighbors,path]=find_path(q,graph_adj);
    clusters=find_clusters(neighbors,path);
    Cq=build_Cq(X,q,prob_params,neighbors,clusters);
    data_compressed=compress(data,Cq);
    X_tilde=prob_solver(prob_params,data_compressed);
    X_block=update_X_block(X_block,X_tilde,q,prob_params,data_compressed,...
    neighbors,clusters,prob_select_sol);
    X=cell2mat(X_block);
    %X=Cq*X_tilde;
    %X_block=mat2cell(X,nbsensors_vec,Q);
    
    
    
    % Global variable.
    %X_block=update_X_block(X_block,X_tilde,q,prob_params,neighbors,clusters,...
    %            prob_select_sol);
    %X=cell2mat(X_block);
    
    % Evaluate the objective.
    if(~isempty(prob_eval))
        f_old=f;
        f=prob_eval(X,data);
        f_seq=[f_seq,f];
    end
    
    if i>0
        norm_diff=[norm_diff,norm(X-X_old,'fro').^2/numel(X)];
    end
    
    %X_block=mat2cell(X,nbsensors_vec,Q);
    
    if(~isempty(X_star) && compare_opt)
        if(~isempty(prob_select_sol))
            X=prob_select_sol(X_star,X,prob_params,data,q);
        end
        norm_err=[norm_err,norm(X-X_star,'fro')^2/norm(X_star,'fro')^2];
        if(plot_dynamic)
            dynamic_plot(X,X_star)
        end
    end
    
    X_block=mat2cell(X,nbsensors_vec,Q);
    
    X_old=X;
    
    i=i+1;
    
    if (tol_f_break && abs(f-f_old)<=tol_f) || (tol_X_break && norm(X-X_old,'fro')<=tol_X)
        break
    end

end

end


