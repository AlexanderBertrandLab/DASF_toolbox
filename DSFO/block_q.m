function Xq=block_q(X,q,nbsensors_vec)

% Function to extract the block of X corresponding to node q.
%
% INPUTS:
% X (nbsensors x Q): Global variable equal to [X1;...;Xq;...;XK].
% q: Updating node.
% nbsensors_vec (nbnodes x nbnodes): Vector containing the number of
%                                    sensors for each node.
%
% OUTPUTS:
% Xq (nbsensors_vec(q) x Q): Block of X corresponding to node q.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    M_q=nbsensors_vec(q);
    row_blk=cumsum(nbsensors_vec);
    row_blk=[0;row_blk(1:end-1)]+1;
    row_blk_q=row_blk(q);
    Xq=X(row_blk_q:row_blk_q+M_q-1,:);
    
end