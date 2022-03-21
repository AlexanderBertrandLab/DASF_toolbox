function Cq=build_Cq(X,q,prob_params,neighbors,clusters)

% Function to construct the transition matrix between the local data and
% variables and the global ones.
%
% INPUTS:
% X (nbsensors x Q): Global variable equal to [X1;...;Xq;...;XK].
% q: Updating node.
% prob_params: Structure related to the problem parameters.
% neighbors: Vector containing the neighbors of node q.
% clusters: Vector of cells. For each neighbor k of q, there is a 
%     corresponding cell with the nodes of the subgraph containing k, 
%     obtained by cutting the link between nodes q and k.
%
% OUTPUTS:
% Cq: Transformation matrix making the transition between local and global
%     data.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be
  
    nbnodes=prob_params.nbnodes;
    nbsensors_vec=prob_params.nbsensors_vec;
    Q=prob_params.Q;
    nbneighbors=length(neighbors);

    ind=0:nbneighbors-1;
    
    Cq=zeros(sum(nbsensors_vec),nbsensors_vec(q)+nbneighbors*Q);
    Cq(:,1:nbsensors_vec(q))=[zeros(sum(nbsensors_vec(1:q-1)),nbsensors_vec(q));...
        eye(nbsensors_vec(q)); zeros(sum(nbsensors_vec(q+1:nbnodes)),nbsensors_vec(q))];
    for k=1:nbneighbors
        ind_k=ind(k);
        for n=1:length(clusters{k})
            clusters_k=clusters{k};
            l=clusters_k(n);
            X_curr=X(sum(nbsensors_vec(1:l-1))+1:sum(nbsensors_vec(1:l)),:);
            Cq(sum(nbsensors_vec(1:l-1))+1:sum(nbsensors_vec(1:l)),...
                nbsensors_vec(q)+ind_k*Q+1:nbsensors_vec(q)+ind_k*Q+Q)=X_curr;
        end
    end
    
end