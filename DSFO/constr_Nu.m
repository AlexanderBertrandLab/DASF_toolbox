function Nu=constr_Nu(neighbors,path)

% Function to obtain clusters of nodes for each neighbor.
%
% INPUTS:
% neighbors: Vector containing the neighbors of node q.
% adj (nbnodes x nbnodes): Adjacency (binary) matrix where K is the number  
%                          of nodes in the network with adj(i,j)=1 if 
%                          i and j are  connected. Otherwise 0. 
%                          adj(i,i)=0.
%
% OUTPUTS:
% Nu: Vector of cells. For each neighbor k of q, there is a corresponding
%     cell with the nodes of the subgraph containing k, obtained by cutting 
%     the link between nodes q and k.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    nbneighbors=length(neighbors);
    Nu=cell(nbneighbors,1);
    for k=1:nbneighbors
        Nu{k}=find(cell2mat(cellfun(@(c) ismember(neighbors(k),c), path, 'uniform', false))==1);
    end
    
end