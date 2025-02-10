function [x,rho_track,norm_star_track]=tro_fdasf(params,data,conv,debug,X_star)

% Function running the distributed trace ratio algorithm on a random
% connected graph.
% INPUTS :
% params : Structure containing the following fields:
%         Q : Number of filters to use (dimension of projected space)
%         nbnodes : Number of nodes in the network
%         nbsensnode : Vector containing the number of sensors for each
%         node
%         nbsens : Sum of the number of sensors for each node (dimension of
%         the network-wide signals). Is equal to sum(nbsensnode)
%         denom_sum : Binary value for determining the trace ratio
%         denominator. If equals 1, the denominator is the sum of the
%         covariance matrices. If 0 not
% data   : Structure containing the following fields:
%         R_first : Covariance matrix in the numerator
%         R_second : Covariance matrix in the denominator
% conv   : Parameters concerning the stopping criteria of the algorithm
%         tol_rho : Tolerance in objective: |rho^(i+1)-rho^(i)|>tol_rho
%         nbiter : Max. nb. of iterations.
% If both values are valid, the algorithm will continue until the stricter
% condition (OR). One of the criteria can be chosen explicitly by
% initializing the other to a negative value.
%
% graph_adj : Adjacency (binary) matrix, with graph_adj(i,j)=1 if i and j
% are connected. Otherwise 0. graph_adj(i,i)=0.
% debug  : If debug equals 1, dynamically plot first projection vector
%          across iterations
% X_star : (Optional) True projection matrix, computed for example with the
%          centralized algorithm. Allows to compare convergence, if it is
%          not provided, there might be a difference in the signs of the
%          columns of the output of this algorithm and W_star
%
% OUTPUTS : 
% x               : Projection matrix
% rho_track       : Sequence of objective values across iterations
% norm_track      : Sequence of ||X^(i+1)-X^(i)||_F^2
% norm_star_track : Sequence of ||X^(i)-X^*||_F^2
%

nbsensors=params.nbsensors;
Q=params.Q;
nbnodes=params.nbnodes;
nbsensors_vec=params.nbsensors_vec;
adj=params.adj;

Ryy=data.Ryy;
Rvv=data.Rvv;

tol_rho=conv.tol_rho;
nbiter=conv.nbiter;

xinit=randn(nbsensors,Q);
xinit=normc(xinit);
rho=tro_obj(xinit,Ryy,Rvv);

current=1;
for k=1:nbnodes
    X{k}=xinit(current:current+nbsensors_vec(k)-1,:);
    x=xinit;
    current=current+nbsensors_vec(k);
end

i=0;
rho_old=rho+1;

rho_track=[];
norm_track=[];
norm_star_track=[];

path=1:nbnodes;
rand_path=path(randperm(length(path)));
X_cell=cell(nbiter,1);

while (tol_rho>0 && abs(rho-rho_old)>tol_rho) || (i<nbiter)
    
    x_old=x;
    
    %q=rem(i,nbnodes)+1;
    q=rand_path(rem(i,nbnodes)+1);
    
    [neighbors,path]=find_path(q,adj);
    
    Nu=constr_Nu(neighbors,path);
    
    C_q=constr_C(X,Q,q,nbsensors_vec,nbnodes,neighbors,Nu);

    [Ryycompressed,Rvvcompressed,Compressor]=compress(Ryy,Rvv,C_q);

    XGnew=comp_XG(Ryycompressed,Rvvcompressed,Compressor,Q,rho);

    rho_old=rho;
    
    if(debug==1)
        xopt_int=tro_aux(Ryy,Rvv,eye(nbsensors),Q,rho);
    end
    
    rho=tro_obj(XGnew,Ryycompressed,Rvvcompressed);
    rho_track=[rho_track,rho];
    
    X=update_X(X,XGnew,Q,q,nbsensors_vec,nbnodes,neighbors,Nu);
    
    x=form_mat(X,x,Q,nbsensors_vec,nbnodes);

    i=i+1;
    X_cell{i}=x;
    
    if(debug==1)
        plot(xopt_int(:,1),'r')
        hold on
        plot(x(:,1),'b')
        ylim([1.2*min(real(xopt_int(:,1))) 1.2*max(real(xopt_int(:,1)))]);
        hold off
        drawnow
    end
    
    if i>1
        norm_track=[norm_track,norm(x-x_old,'fro').^2/numel(x)];
    end

end

for l=1:Q
    if sum(sum((X_star(:,l)-x(:,l)).^2))>sum(sum((-X_star(:,l)-x(:,l)).^2))
        X_star(:,l)=-X_star(:,l);
    end
end

norm_star_track=[];
for i=1:length(X_cell)
    norm_star_track(i)=norm(X_cell{i}-X_star,'fro')^2/norm(X_star,'fro')^2;
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


function C_q=constr_C(X,Q,q,nbsensnode,nbnodes,neighbors,Nu)
   
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
                nbsensnode(q)+ind_k*Q+1:nbsensnode(q)+ind_k*Q+Q)=X{l};
        end
    end
    
end

function [Ryycompressed,Rvvcompressed,Compressor]=compress(Ryy,Rvv,C_q)

    Ryycompressed=C_q'*Ryy*C_q;
    Ryycompressed=make_sym(Ryycompressed);
    
    Rvvcompressed=C_q'*Rvv*C_q;
    Rvvcompressed=make_sym(Rvvcompressed);
    
    Compressor=C_q'*C_q;
    Compressor=make_sym(Compressor);

end

function XG=comp_XG(Ryycompressed,Rvvcompressed,Compressor,Q,rho)
    XG=tro_aux(Ryycompressed,Rvvcompressed,Compressor,Q,rho);
end

function X=update_X(X,XGnew,Q,q,nbsensnode,nbnodes,neighbors,Nu)

    Xqold=X{q};
    X{q}=XGnew(1:nbsensnode(q),:);
    
    for l=1:Q
        if sum(sum((Xqold(:,l)-X{q}(:,l)).^2))>sum(sum((-Xqold(:,l)-X{q}(:,l)).^2))
            X{q}(:,l)=-X{q}(:,l);
            XGnew(:,l)=-XGnew(:,l);
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
        X{l}=X{l}*XGnew(start_r:stop_r,:);
    end
    for l=q+1:nbnodes
        for k=1:nb_neighbors
            if ~isempty(find(Nu{k} == l))
                start_r=nbsensnode(q)+ind(k)*Q+1;
                stop_r=nbsensnode(q)+ind(k)*Q+Q;
            end
        end
        X{l}=X{l}*XGnew(start_r:stop_r,:);
    end

end

function x=form_mat(X,x,Q,nbsensnode,nbnodes)
    current=1;
    for l=1:nbnodes
        x(current:current+nbsensnode(l)-1,:)=X{l}(:,1:Q);
        current=current+nbsensnode(l);
    end
end

