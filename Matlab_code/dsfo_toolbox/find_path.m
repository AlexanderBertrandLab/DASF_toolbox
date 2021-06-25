function [neighbors,path]=find_path(q,adj)

% Function finding the neighbors of node q and the shortest path to other
% every other node in the network.
%
% INPUTS:
% q: Source node.
% adj (nbnodes x nbnodes): Adjacency (binary) matrix where K is the number  
%                          of nodes in the network with adj(i,j)=1 if 
%                          i and j are  connected. Otherwise 0. 
%                          adj(i,i)=0.
%
% OUTPUTS:
% neighbors: Vector containing the neighbors of node q.
% path (nbnodes x 1): Cell of vectors containining at index k the shortest 
%                     path from node q to node k.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    [dist,path]=shortest_path(q,adj);
    neighbors=find(cell2mat(cellfun(@(c) length(c), path, 'uniform', false))==2);
    neighbors=sort(neighbors);
    
end