function [Ufull,Vfull]=create_data(nbsamples,nbsensnode,nbnodes,Q,D)
% Function to create synthetic signals.
% INPUTS:
% nbsamples: Number of time samples.
% nbsensnode (nbsamples x 1): Vector of length nbnodes containing the number of channels per node.
% nbnodes: Number of nodes in the network.
% Q: Projection dimension.
%
% CONSTANTS:
% D: Dimension of the latent random process.
% noisepower: Variance of the white noise.
% signalvar: Variance of the source and interference signals.
%
% OUTPUTS:
% Ufull, Vfull (nbsamples x sum(nbsensnode)): Matrices containing
%               the desired and interfering signals respectively.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be
   
noisepower=0.1; 
signalvar=0.5;

rng('shuffle');
d=randn(nbsamples,Q); %random latent process
d=sqrt(signalvar)./(sqrt(var(d))).*(d-ones(nbsamples,1)*mean(d));
s=randn(nbsamples,D-Q); %same random latent process, but new observations
s=sqrt(signalvar)./(sqrt(var(s))).*(s-ones(nbsamples,1)*mean(s));

for k=1:nbnodes
    Ainit{k}=rand(Q,nbsensnode(k))-0.5;
    Binit{k}=rand(D-Q,nbsensnode(k))-0.5;
    noise{k}=randn(nbsamples,nbsensnode(k)); 
    noise{k}=sqrt(noisepower)./sqrt(var(noise{k})).*(noise{k}-ones(nbsamples,1)*mean(noise{k}));
end

teller=0;

for k=1:nbnodes
    V{k}=s*Binit{k}+noise{k};
    U{k}=d*Ainit{k}+V{k};

    Ufull(1:nbsamples,teller+1:teller+nbsensnode(k))=U{k};
    Vfull(1:nbsamples,teller+1:teller+nbsensnode(k))=V{k};
    teller=teller+nbsensnode(k);
end

end