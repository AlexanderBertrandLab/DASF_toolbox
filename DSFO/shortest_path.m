function [dist,path]=shortest_path(adj,q)

% Function computing the shortest path distance between a source node and
% all nodes in the network using Dijkstra's method.
% Note: This implementation is only for graphs for which the weight at each
% edge is equal to 1.
% INPUTS:
% adj (K x K): Adjacency (binary) matrix where K is the number of nodes in 
%           the network with adj(i,j)=1 if i and j are connected. 
%           Otherwise 0. 
%           adj(i,i)=0.
% q: Source node
%
% OUTPUTS:
% dist (K x 1): Distances between the source node and other nodes.
% path (1 x K): Cell containing the shortest paths from the source node to
% the other nodes.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    K=size(adj,1);
    dist=Inf(K,1);
    dist(q)=0;
   
    visited=[];
    pred=zeros(1,K);
    unvisited=setdiff([1:K],visited);
    path=cell(1,K);

    while(length(visited)<K)
        I=find(dist==min(dist(unvisited)));
        I=I';
       
        for ind=I
            visited=[visited,ind];
            unvisited=setdiff([1:K],visited);
            neighbors_i=find(adj(ind,:)==1);
            for m=intersect(neighbors_i,unvisited)
                if(dist(ind)+1<dist(m))
                    dist(m)=dist(ind)+1;
                    pred(m)=ind;
                end
            end
        end
        
    end
    
    for k=1:K
        jmp=k;
        path_k=[k];
        while(jmp~=q)
            jmp=pred(jmp);
            path_k=[jmp,path_k];
        end
        path(k)={path_k};
    end
    
end