function [Y,B,C,d]=create_data(nbsensors,nbsamples,Q)

    signalvar=0.5;
    Mixvar=0.2;
    noisepower=0.2;
    nbsources=Q;
    
    rng('shuffle');
    S=randn(nbsources,nbsamples);
    S=sqrt(signalvar)./(sqrt(var(S,0,2))).*(S-mean(S,2)*ones(1,nbsamples));

    Mix=sqrt(Mixvar)*randn(nbsensors,nbsources);
    noise=randn(nbsensors,nbsamples);
    noise=sqrt(noisepower)./sqrt(var(noise,0,2)).*(noise...
            -mean(noise,2)*ones(1,nbsamples));

    Y=Mix*S+noise;
    
    B=randn(nbsensors,Q);
    C=randn(nbsensors,Q);
    
    Ryy=1/nbsamples*conj(Y*Y');
    Ryy=make_sym(Ryy);
    invR=inv(Ryy);
    invR=make_sym(invR);
    
    threshold1=(trace(C'*invR*B)-sqrt(trace(B'*invR*B)*trace(C'*invR*C)))/2;
    threshold2=(trace(C'*invR*B)+sqrt(trace(B'*invR*B)*trace(C'*invR*C)))/2;
    
    d=threshold2+randi(1000);

end