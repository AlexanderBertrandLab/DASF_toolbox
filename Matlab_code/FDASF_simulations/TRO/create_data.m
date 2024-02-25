function [Y,V]=create_data(nbsensors,nbsamples)
    
    noisepower=0.1;
    signalvar=0.5;
    nbsources=2;
    latent_dim=4;

    rng('shuffle');
    D=randn(nbsources,nbsamples);
    D=sqrt(signalvar)./(sqrt(var(D,0,2))).*(D-mean(D,2)*ones(1,nbsamples));
    S=randn(latent_dim-nbsources,nbsamples);
    S=sqrt(signalvar)./(sqrt(var(S,0,2))).*(S-mean(S,2)*ones(1,nbsamples));
    
    A=sqrt(0.1)*randn(nbsensors,nbsources);
    B=sqrt(0.1)*randn(nbsensors,latent_dim-nbsources);
    noise=randn(nbsensors,nbsamples);
    noise=sqrt(noisepower)./sqrt(var(noise,0,2)).*(noise...
            -mean(noise,2)*ones(1,nbsamples));
        
    V=B*S+noise;
    Y=A*D+V;

end