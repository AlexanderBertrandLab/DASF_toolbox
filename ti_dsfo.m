function [X,f_track,norm_track,norm_star_track]=ti_dsfo(prob_params,data,obj_eval,prob_solver,conv,debug,X_star)

% Function running the TI-DSFO for a given problem.
% INPUTS :
% prob_params : Structure containing the following fields:
%            Q : Number of filters to use (dimension of projected space)
%            nbnodes : Number of nodes in the network.
%            nbsensnode : Vector containing the number of sensors for each
%                         node.
%            nbsens : Sum of the number of sensors for each node 
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
% conv        : Parameters concerning the stopping criteria of the algorithm
%            tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
%            nbiter : Max. nb. of iterations.
% If both values are valid, the algorithm will continue until the stricter
% condition (OR). One of the criteria can be chosen explicitly by
% initializing the other to a negative value.
%
% debug       : If debug equals 1, dynamically plot first projection vector
%               across iterations
% X_star      : (Optional) True projection matrix, computed for example
%               with the centralized algorithm. Allows to compare 
%               convergence, if it is not provided, there might be a 
%               difference in the signs of the columns of the output of 
%               this algorithm and X_star.
%
% OUTPUTS : 
% X               : Projection matrix
% f_track         : Sequence of objective values across iterations
% norm_track      : Sequence of ||X^(i+1)-X^(i)||_F^2
% norm_star_track : Sequence of ||X^(i)-X^*||_F^2/||X^*||_F^2
%


Q=prob_params.Q;
nbsens=prob_params.nbsens;
nbnodes=prob_params.nbnodes;
nbsensnode=prob_params.nbsensnode;
adj=prob_params.graph_adj;
sgn_sync=prob_params.sgn_sync;
efficient=prob_params.efficient;

tol_f=conv.tol_f;
nbiter=conv.nbiter;

Xinit=randn(nbsens,Q);
Xinit=normc(Xinit);
f=obj_eval(Xinit,data);

inter_node=1;
X_cell=cell(nbnodes,1);
for k=1:nbnodes
    X_cell{k}=Xinit(inter_node:inter_node+nbsensnode(k)-1,:);
    X=Xinit;
    inter_node=inter_node+nbsensnode(k);
end

i=0;
f_old=f+1;

f_track=[];
norm_track=[];
norm_star_track=[];

path=1:nbnodes;
rand_path=path(randperm(length(path)));

while (tol_f>0 && abs(f-f_old)>tol_f) || (i<nbiter)
    
    X_old=X;
    
    q=rand_path(rem(i,nbnodes)+1);
    
    [neighbors,path]=find_path(q,adj);
    
    Nu=constr_Nu(neighbors,path);
    
    C_q=constr_C(X_cell,Q,q,nbsensnode,nbnodes,neighbors,Nu);

    data_compressed=compress(data,C_q);

    X_tilde=comp_X_tilde(prob_params,data_compressed,prob_solver);
    
    f_old=f;
    f=obj_eval(X_tilde,data_compressed);
    f_track=[f_track,f];
    
    if efficient==0
        X_cell=update_X(X_cell,X_tilde,Q,q,nbsensnode,nbnodes,neighbors,Nu);
    else
        X_cell=update_X_efficient(X_cell,X_tilde,Q,q,nbsensnode,nbnodes,neighbors,Nu);
    end
    
    X=form_mat(X_cell,X,Q,nbsensnode,nbnodes);

    i=i+1;
    
    if(debug==1)
        for l=1:Q
            if sum(sum((X_star(:,l)-X(:,l)).^2))>sum(sum((-X_star(:,l)-X(:,l)).^2))
                X(:,l)=-X(:,l);
            end
        end
    
        plot(X_star(:,1),'r')
        hold on
        plot(X(:,1),'b')
        ylim([1.2*min(real(X_star(:,1))) 1.2*max(real(X_star(:,1)))]);
        hold off
        drawnow
    end
    
    if i>1
        norm_track=[norm_track,norm(X-X_old,'fro').^2/numel(X)];
    end
    
    if nargin>5
        
        if(sgn_sync==1)
            for l=1:Q
                if sum(sum((X_star(:,l)-X(:,l)).^2))>sum(sum((-X_star(:,l)-X(:,l)).^2))
                    X(:,l)=-X(:,l);
                end
            end
        end

        norm_star_track=[norm_star_track,norm(X-X_star,'fro')^2/norm(X_star,'fro')^2];
    end

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


function C_q=constr_C(X_cell,Q,q,nbsensnode,nbnodes,neighbors,Nu)
   
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

function data_compressed=compress(data,C_q)
    
    Y_cell=data.Y_cell;
    B_cell=data.B_cell;
    Gamma_cell=data.Gamma_cell;
    
    data_compressed=struct;

    if(~isempty(data.Y_cell))
        S=length(Y_cell);
        Y_cell_compressed=cell(S,1);
        for s=1:S
            Y_cell_compressed{s}=C_q'*Y_cell{s};
        end
        data_compressed.Y_cell=Y_cell_compressed;
    else
        data_compressed.Y_cell={};
    end
    
    if(~isempty(data.B_cell))
        P=length(B_cell);
        B_cell_compressed=cell(P,1);
        for p=1:P
            B_cell_compressed{p}=C_q'*B_cell{p};
        end
        data_compressed.B_cell=B_cell_compressed;
    else
        data_compressed.B_cell={};
    end
    
    if(~isempty(data.Gamma_cell))
        D=length(Gamma_cell);
        Gamma_cell_compressed=cell(D,1);
        for d=1:D
            Gamma_cell_compressed{d}=C_q'*Gamma_cell{d}*C_q;
            Gamma_cell_compressed{d}=make_sym(Gamma_cell_compressed{d});
        end
        data_compressed.Gamma_cell=Gamma_cell_compressed;
    else
        data_compressed.Gamma_cell={};
    end
    
end

function X_tilde=comp_X_tilde(prob_params,data_compressed,prob_solver)

    [X_tilde,~]=prob_solver(prob_params,data_compressed);
    
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
