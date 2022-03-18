function X_block_upd=update_X_block(X_block,X_tilde,q,prob_params,neighbors,...
    Nu,prob_select_sol)

% Function to update the cell containing the blocks of X for each
% corresponding node.
%
% INPUTS:
% X_block (nbnodes x 1): Vector of cells containing the current blocks Xk^i
% at each cell k, where X=[X1;...;Xk;...;XK].
% X_tilde: Solution of the local problem at the current updating node.
% q: Updating node.
% prob_params: Structure related to the problem parameters.
% neighbors: Vector containing the neighbors of node q.
% Nu: Vector of cells. For each neighbor k of q, there is a corresponding
%     cell with the nodes of the subgraph containing k, obtained by cutting 
%     the link between nodes q and k.
% prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.
%
% OUTPUTS:
% X_block_upd (nbnodes x 1): Vector of cells with updated block Xk^(i+1).
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be


    nbnodes=prob_params.nbnodes;
    nbsensors_vec=prob_params.nbsensors_vec;
    Q=prob_params.Q;
    nbneighbors=length(neighbors);
    
    if(~isempty(prob_select_sol))
        Xq_old=X_block{q};
        X_tilde_old=[Xq_old;repmat(eye(Q),nbneighbors,1)];
        X_tilde=prob_select_sol(X_tilde_old,X_tilde,nbsensors_vec,q);
    end
    
    X_block_upd=cell(nbnodes,1);
    X_block_upd{q}=X_tilde(1:nbsensors_vec(q),:);
    
    ind=0:nbneighbors-1;
    
    for l=1:q-1
        for k=1:nbneighbors
            if ~isempty(find(Nu{k}==l))
                start_r=nbsensors_vec(q)+ind(k)*Q+1;
                stop_r=nbsensors_vec(q)+ind(k)*Q+Q;
            end
        end
        X_block_upd{l}=X_block{l}*X_tilde(start_r:stop_r,:);
    end
    for l=q+1:nbnodes
        for k=1:nbneighbors
            if ~isempty(find(Nu{k}==l))
                start_r=nbsensors_vec(q)+ind(k)*Q+1;
                stop_r=nbsensors_vec(q)+ind(k)*Q+Q;
            end
        end
        X_block_upd{l}=X_block{l}*X_tilde(start_r:stop_r,:);
    end

end
