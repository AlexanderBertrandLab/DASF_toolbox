function [x,rho_track,norm_track,norm_star_track,ind_dasf]=dasf_rtls(params,data,conv,debug,X_star)

nbsens=params.nbsens;
Q=params.Q;
nbnodes=params.nbnodes;
nbsensnode=params.nbsensnode;

Ryy=data.Ryy;
ryd=data.ryd;
rdd=data.rdd;
LL=data.LL;
delta=data.delta;

tol_rho=conv.tol_rho;
nbiter=conv.nbiter;

xinit=randn(nbsens,Q);
xinit=normc(xinit);
rho=tls_obj(Ryy,ryd,rdd,xinit,eye(nbsens));

current=1;
for k=1:nbnodes
    X{k}=xinit(current:current+nbsensnode(k)-1,:);
    x=xinit;
    current=current+nbsensnode(k);
end

i=0;
rho_old=rho+1;

rho_track=[];
norm_track=[];
norm_star_track=[];
ind_dasf=[];

adj=params.adj;

path=1:nbnodes;
rand_path=path(randperm(length(path)));

while (tol_rho>0 && abs(rho-rho_old)>tol_rho) || (i<nbiter)
    
    x_old=x;
    
    q=rand_path(rem(i,nbnodes)+1);
    
    [neighbors,path]=find_path(q,adj);
    
    Nu=constr_Nu(neighbors,path);
    
    C_k=constr_C(X,Q,q,nbsensnode,nbnodes,neighbors,Nu);
    
    [Ryyc,rydc,LLc,Compressor]=compress(Ryy,ryd,LL,C_k);
    
    [XGnew,nb_ind]=comp_XG(Ryyc,rydc,rdd,LLc,Compressor,delta);

    rho_old=rho;
    
    if(debug==1)
        xopt_int=tls_aux(Ryy,ryd,rdd,LL,eye(nbsens),delta,rho);
    end
    
    rho=tls_obj(Ryyc,rydc,rdd,XGnew,Compressor);
    rho_track=[rho_track,rho];
    
    X=update_X(X,XGnew,Q,q,nbsensnode,nbnodes,neighbors,Nu);
    
    x=form_mat(X,x,Q,nbsensnode,nbnodes);

    i=i+1;
    
    ind_dasf=[ind_dasf,nb_ind];
    
    if(debug==1)    
        plot(xopt_int(:,1),'r')
        hold on
        plot(x(:,1),'b')
        ylim([1.2*min(real(xopt_int(:,1))) 1.2*max(real(xopt_int(:,1)))]);
        hold off
        drawnow
    end
    
    if i>1
        norm_track=[norm_track,norm(x-x_old,'fro')^2/numel(x)];
    end
    
    if nargin>4
        norm_star_track=[norm_star_track,norm(x-X_star,'fro')^2/norm(X_star)^2];
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


function C_k=constr_C(X,Q,q,nbsensnode,nbnodes,neighbors,Nu)
   
    nb_neighbors=length(neighbors);

    ind=0:nb_neighbors-1;
    
    C_k=zeros(sum(nbsensnode),nbsensnode(q)+nb_neighbors*Q);
    C_k(:,1:nbsensnode(q))=[zeros(sum(nbsensnode(1:q-1)),nbsensnode(q));...
        eye(nbsensnode(q)); zeros(sum(nbsensnode(q+1:nbnodes)),nbsensnode(q))];
    for k=1:nb_neighbors
        ind_k=ind(k);
        for n=1:length(Nu{k})
            Nu_k=Nu{k};
            l=Nu_k(n);
            C_k(sum(nbsensnode(1:l-1))+1:sum(nbsensnode(1:l)),...
                nbsensnode(q)+ind_k*Q+1:nbsensnode(q)+ind_k*Q+Q)=X{l};
        end
    end
    
end

function [Ryyc,rydc,LLc,Compressor]=compress(Ryy,ryd,LL,C_k)

    Ryyc=C_k'*Ryy*C_k;
    rydc=C_k'*ryd;
    LLc=C_k'*LL*C_k;
    Compressor=C_k'*C_k;
    Ryyc=make_sym(Ryyc);
    LLc=make_sym(LLc);
    Compressor=make_sym(Compressor);

end

function [XG,nb_ind]=comp_XG(Ryyc,rydc,rdd,LLc,Compressor,delta)

    [XG,~,nb_ind]=rtls_solver(Ryyc,rydc,rdd,LLc,Compressor,delta);
    
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

