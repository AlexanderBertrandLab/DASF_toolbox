function [x,rho_track,norm_star_track]=qol_fdasf(params,data,conv,debug,X_star)

nbsensors=params.nbsensors;
Q=params.Q;
nbnodes=params.nbnodes;
nbsensors_vec=params.nbsensors_vec;
adj=params.adj;

Ryy=data.Ryy;
B=data.B;
C=data.C;
d=data.d;

tol_rho=conv.tol_rho;
nbiter=conv.nbiter;

xinit=randn(nbsensors,Q);
xinit=normc(xinit);
rho=qol_obj(xinit,Ryy,B,C,d);

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

while (tol_rho>0 && abs(rho-rho_old)>tol_rho) || (i<nbiter)
    
    w_old=x;
    
    q=rand_path(rem(i,nbnodes)+1);
    
    [neighbors,path]=find_path(q,adj);
    
    Nu=constr_Nu(neighbors,path);
    
    C_q=constr_C(X,Q,q,nbsensors_vec,nbnodes,neighbors,Nu);

    [Ryycompressed,Bcompressed,Ccompressed]=compress(Ryy,B,C,C_q);

    XGnew=comp_XG(Ryycompressed,Bcompressed,Ccompressed,rho);

    rho_old=rho;
    
    if(debug==1)
        xopt_int=qol_aux(Ryy,B,C,rho);
    end
    
    rho=qol_obj(XGnew,Ryycompressed,Bcompressed,Ccompressed,d);
    rho_track=[rho_track,rho];
    
    X=update_X(X,XGnew,Q,q,nbsensors_vec,nbnodes,neighbors,Nu);
    
    x=form_mat(X,x,Q,nbsensors_vec,nbnodes);

    i=i+1;
    
    if(debug==1)
        plot(xopt_int(:,1),'r')
        hold on
        plot(x(:,1),'b')
        ylim([1.2*min(real(xopt_int(:,1))) 1.2*max(real(xopt_int(:,1)))]);
        hold off
        drawnow
    end
    
    if i>1
        norm_track=[norm_track,norm(x-w_old,'fro').^2/numel(x)];
    end
    
    if nargin>4
        norm_star_track=[norm_star_track,norm(x-X_star,'fro')^2/norm(X_star,'fro')^2];
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

function [Ryycompressed,Bcompressed,Ccompressed]=compress(Ryy,B,C,C_q)

    Ryycompressed=C_q'*Ryy*C_q;
    Ryycompressed=make_sym(Ryycompressed);

    Bcompressed=C_q'*B;
    Ccompressed=C_q'*C;

end

function XG=comp_XG(Ryycompressed,Bcompressed,Ccompressed,rho)

    XG=qol_aux_solver(Ryycompressed,Bcompressed,Ccompressed,rho);
    
end

function X=update_X(X,XGnew,Q,q,nbsensnode,nbnodes,neighbors,Nu)

    Xqold=X{q};
    X{q}=XGnew(1:nbsensnode(q),:);
    
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

