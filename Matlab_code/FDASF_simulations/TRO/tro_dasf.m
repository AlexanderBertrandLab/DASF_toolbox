function [x,rho_track,norm_star_track,iter_count_vec]=tro_dasf(params,data,conv,debug,X_star)

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
iter_count_vec=[];

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

    [XGnew,iter_count]=comp_XG(Ryycompressed,Rvvcompressed,Compressor,Q);

    rho_old=rho;
    
    if(debug==1)
        xopt_int=tro_aux(Ryy,Rvv,eye(nbsensors),Q,rho);
    end
    
    rho=tro_obj(XGnew,Ryycompressed,Rvvcompressed);
    rho_track=[rho_track,rho];
    iter_count_vec=[iter_count_vec,iter_count];
    
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

function [XG,iter_count]=comp_XG(Ryycompressed,Rvvcompressed,Compressor,Q)
    [XG,iter_count,~]=tro(Ryycompressed,Rvvcompressed,Compressor,Q);
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

