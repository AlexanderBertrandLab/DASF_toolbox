function [X,f_diff,norm_diff,norm_err]=dsfo(data,prob_params,conv,...
    obj_eval,prob_solver,prob_resolve_uniqueness)

% Function running the TI-DSFO for a given problem.
% INPUTS :
% prob_params : Structure containing the following fields:
%            Q : Number of filters to use (dimension of projected space)
%            nbnodes : Number of nodes in the network.
%            nbsensors_vec : Vector containing the number of sensors for each
%                         node.
%            nbsensors : Sum of the number of sensors for each node 
%                     (dimension of the network-wide signals). Is equal to
%                     sum(nbsensnode).
%            graph_adj : Adjacency (binary) matrix, with graph_adj(i,j)=1  
%                        if i and j are connected. Otherwise 0. 
%                        graph_adj(i,i)=0.
% data        : Structure containing the following fields:
%            Y_cell : Cell containing S matrices of size M x N representing
%                     the signals.
%            B_cell : Cell containing P matrices of size M x L(p), 1<=p<=P,
%                     representing the parameters.
%            Gamma_cell : Cell containing D matrices of size M x M
%                         representing the quadratic parameters.
%            Glob_Const_cell : Cell containing the global constants which
%                         are not filtered through X.
% conv        : Parameters concerning the stopping criteria of the algorithm
%            tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
%            nbiter : Max. nb. of iterations.
% If both values are valid, the algorithm will continue until the stricter
% condition (OR). One of the criteria can be chosen explicitly by
% initializing the other to a negative value.
%
% obj_eval    : Function evaluating the objective of the problem
%
% prob_solver : Function solving the centralozed problem
%
% prob_resolve_uniqueness : (Optional) Function resolving the uniqueness
%                           ambiguity
%
%
% OUTPUTS : 
% X               : Projection matrix
% f_diff          : Sequence of objective values across iterations
% norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q)
% norm_err        : Sequence of ||X^(i)-X^*||_F^2/||X^*||_F^2
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
compare_opt=prob_params.compare_opt;
plot_dynamic=prob_params.plot_dynamic;
X_star=prob_params.X_star;

tol_f=conv.tol_f;
nbiter=conv.nbiter;

X=randn(nbsensors,Q);
X=normc(X);
f=obj_eval(X,data);

i=0;
f_old=f+1;

f_diff=[];
norm_diff=[];
norm_err=[];

path=1:nbnodes;
% Random updating order
rand_path=path(randperm(length(path)));

while (tol_f>0 && abs(f-f_old)>tol_f) || (i<nbiter)
    
    % Select updating node
    q=rand_path(rem(i,nbnodes)+1);
    
    % Prune the network
    % Find shortest path
    [neighbors,path]=find_path(q,adj);
    
    % Neighborhood clusters
    Nu=constr_Nu(neighbors,path);
    
    % Global - local transition matrix
    C_q=constr_C(X,Q,q,nbsensors_vec,nbnodes,neighbors,Nu);

    % Compute compressed data
    data_compressed=compress(data,C_q);

    % Compute the local variable
    X_tilde=comp_X_tilde(data_compressed,prob_params,prob_solver);
    
    % Evaluate objective
    f_old=f;
    f=obj_eval(X_tilde,data_compressed);
    f_diff=[f_diff,f];
    
    % Global variable
    X=C_q*X_tilde;
    
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

function [neighbors,path]=find_path(q,adj)

    [dist,path]=shortest_path(adj,q);
    neighbors=find(cell2mat(cellfun(@(c) length(c), path, 'uniform', false))==2);
    neighbors=sort(neighbors);
    
end

function Nu=constr_Nu(neighbors,path)

    nb_neighbors=length(neighbors);
    Nu=cell(nb_neighbors,1);
    for k=1:nb_neighbors
        Nu{k}=find(cell2mat(cellfun(@(c) ismember(neighbors(k),c), path, 'uniform', false))==1);
    end
    
end

function C_q=constr_C(x,Q,q,nbsensors_vec,nbnodes,neighbors,Nu)
  
    nb_neighbors=length(neighbors);

    ind=0:nb_neighbors-1;
    
    C_q=zeros(sum(nbsensors_vec),nbsensors_vec(q)+nb_neighbors*Q);
    C_q(:,1:nbsensors_vec(q))=[zeros(sum(nbsensors_vec(1:q-1)),nbsensors_vec(q));...
        eye(nbsensors_vec(q)); zeros(sum(nbsensors_vec(q+1:nbnodes)),nbsensors_vec(q))];
    for k=1:nb_neighbors
        ind_k=ind(k);
        for n=1:length(Nu{k})
            Nu_k=Nu{k};
            l=Nu_k(n);
            X_curr=x(sum(nbsensors_vec(1:l-1))+1:sum(nbsensors_vec(1:l)),:);
            C_q(sum(nbsensors_vec(1:l-1))+1:sum(nbsensors_vec(1:l)),...
                nbsensors_vec(q)+ind_k*Q+1:nbsensors_vec(q)+ind_k*Q+Q)=X_curr;
        end
    end
    
end

function data_compressed=compress(data,C_q)
    
    Y_cell=data.Y_cell;
    B_cell=data.B_cell;
    Gamma_cell=data.Gamma_cell;
    Glob_Const_cell=data.Glob_Const_cell;
    
    data_compressed=struct;

    if(~isempty(data.Y_cell))
        nbsignals=length(Y_cell);
        Y_cell_compressed=cell(nbsignals,1);
        for ind=1:nbsignals
            Y_cell_compressed{ind}=C_q'*Y_cell{ind};
        end
        data_compressed.Y_cell=Y_cell_compressed;
    else
        data_compressed.Y_cell={};
    end
    
    if(~isempty(data.B_cell))
        nbparams=length(B_cell);
        B_cell_compressed=cell(nbparams,1);
        for ind=1:nbparams
            B_cell_compressed{ind}=C_q'*B_cell{ind};
        end
        data_compressed.B_cell=B_cell_compressed;
    else
        data_compressed.B_cell={};
    end
    
    if(~isempty(data.Gamma_cell))
        nbquadr=length(Gamma_cell);
        Gamma_cell_compressed=cell(nbquadr,1);
        for ind=1:nbquadr
            Gamma_cell_compressed{ind}=C_q'*Gamma_cell{ind}*C_q;
            Gamma_cell_compressed{ind}=make_sym(Gamma_cell_compressed{ind});
        end
        data_compressed.Gamma_cell=Gamma_cell_compressed;
    else
        data_compressed.Gamma_cell={};
    end
    
    if(~isempty(data.Glob_Const_cell))
        data_compressed.Glob_Const_cell=Glob_Const_cell;
    else
        data_compressed.Glob_Const_cell={};
    end
    
end

function X_tilde=comp_X_tilde(data_compressed,prob_params,prob_solver)

    % Solve the local problem using the algorithm for the global problem
    % using compressed data
    [X_tilde,~]=prob_solver(data_compressed,prob_params);
    
end

function Xq=block_q(X,q,nbsensors_vec)

    M_q=nbsensors_vec(q);
    row_blk=cumsum(nbsensors_vec);
    row_blk=[0;row_blk(1:end-1)]+1;
    row_blk_q=row_blk(q);
    Xq=X(row_blk_q:row_blk_q+M_q-1,:);
    
end

function dynamic_plot(X,X_star)

    plot(X_star(:,1),'r')
    hold on
    plot(X(:,1),'b')
    ylim([1.2*min(real(X_star(:,1))) 1.2*max(real(X_star(:,1)))]);
    hold off
    drawnow
    
end


%
% Old functions
%

function C_q=constr_C_cell(X_cell,Q,q,nbsensnode,nbnodes,neighbors,Nu)
   
    nb_neighbors=length(neighbors);

    ind=0:nb_neighbors-1;
    
    C_q=zeros(sum(nbsensnode),nbsensnode(q)+nb_neighbors*Q);
    C_q(:,1:nbsensnode(q))=[zeros(sum(nbsensnode(1:q-1)),nbsensnode(q));...
        eye(nbsensnode(q)); zeros(sum(nbsensnode(q+1:nbnodes)),nbsensnode(q))];
    for k=1:nb_neighbors
        ind_k=ind(k);
        for n=1:length(Nu{k})
            Nu_k=Nu{k};
            l=Nu_k(n);
            C_q(sum(nbsensnode(1:l-1))+1:sum(nbsensnode(1:l)),...
                nbsensnode(q)+ind_k*Q+1:nbsensnode(q)+ind_k*Q+Q)=X_cell{l};
        end
    end
    
end

function X_cell=update_X_efficient(X_cell,X_tilde,Q,q,nbsensnode,nbnodes,neighbors,Nu)
 
    Xqold=X_cell{q};
    X_cell{q}=X_tilde(1:nbsensnode(q),:);
    
    for l=1:Q
        if sum(sum((Xqold(:,l)-X_cell{q}(:,l)).^2))>sum(sum((-Xqold(:,l)-X_cell{q}(:,l)).^2))
            X_cell{q}(:,l)=-X_cell{q}(:,l);
            X_tilde(:,l)=-X_tilde(:,l);
        end
    end
    
    nb_neighbors=length(neighbors);
    ind=0:nb_neighbors-1;
    
    for l=1:q-1
        for k=1:nb_neighbors
            if ~isempty(find(Nu{k} == l))
                start_r=nbsensnode(q)+ind(k)*Q+1;
                stop_r=nbsensnode(q)+ind(k)*Q+Q;
            end
        end
        G=diag(sign(diag(X_tilde(start_r:stop_r,:))));
        X_cell{l}=X_cell{l}*G;
    end
    for l=q+1:nbnodes
        for k=1:nb_neighbors
            if ~isempty(find(Nu{k} == l))
                start_r=nbsensnode(q)+ind(k)*Q+1;
                stop_r=nbsensnode(q)+ind(k)*Q+Q;
            end
        end
        G=diag(sign(diag(X_tilde(start_r:stop_r,:))));
        X_cell{l}=X_cell{l}*G;
    end
    
    
end

function X_cell=update_X(X_cell,X_tilde,Q,q,nbsensnode,nbnodes,neighbors,Nu)

    Xqold=X_cell{q};
    X_cell{q}=X_tilde(1:nbsensnode(q),:);
    
    for l=1:Q
        if sum(sum((Xqold(:,l)-X_cell{q}(:,l)).^2))>sum(sum((-Xqold(:,l)-X_cell{q}(:,l)).^2))
            X_cell{q}(:,l)=-X_cell{q}(:,l);
            X_tilde(:,l)=-X_tilde(:,l);
        end
    end
    
    nb_neighbors=length(neighbors);
    ind=0:nb_neighbors-1;
    
    for l=1:q-1
        for k=1:nb_neighbors
            if ~isempty(find(Nu{k} == l))
                start_r=nbsensnode(q)+ind(k)*Q+1;
                stop_r=nbsensnode(q)+ind(k)*Q+Q;
            end
        end
        X_cell{l}=X_cell{l}*X_tilde(start_r:stop_r,:);
    end
    for l=q+1:nbnodes
        for k=1:nb_neighbors
            if ~isempty(find(Nu{k} == l))
                start_r=nbsensnode(q)+ind(k)*Q+1;
                stop_r=nbsensnode(q)+ind(k)*Q+Q;
            end
        end
        X_cell{l}=X_cell{l}*X_tilde(start_r:stop_r,:);
    end

end

function X=form_mat(X_cell,X,Q,nbsensnode,nbnodes)

    inter_node=1;
    for l=1:nbnodes
        X(inter_node:inter_node+nbsensnode(l)-1,:)=X_cell{l}(:,1:Q);
        inter_node=inter_node+nbsensnode(l);
    end

end
