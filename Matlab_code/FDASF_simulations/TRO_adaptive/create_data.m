function [Y,V]=create_data(nbsensors,nbsamples,Q,nbiter)
    
    noisepower=0.1;
    signalvar=0.5;
    nbsources=2;
    latent_dim=4;

    rng('shuffle');
    D=randn(nbsources,nbsamples);
    D=sqrt(signalvar)./(sqrt(var(D,0,2))).*(D-mean(D,2)*ones(1,nbsamples));
    S=randn(latent_dim-nbsources,nbsamples);
    S=sqrt(signalvar)./(sqrt(var(S,0,2))).*(S-mean(S,2)*ones(1,nbsamples));
    
    A=sqrt(0.5)*randn(nbsensors,nbsources);
    B=sqrt(0.1)*randn(nbsensors,latent_dim-nbsources);
    noise=randn(nbsensors,nbsamples);
    noise=sqrt(noisepower)./sqrt(var(noise,0,2)).*(noise...
            -mean(noise,2)*ones(1,nbsamples));
        
    Delta=sqrt(0.01)*randn(nbsensors,nbsources);
    %Delta=Delta*0.05/norm(Delta,'fro')*norm(A,'fro');
    A2=A+Delta;
    
    DeltaB=sqrt(0.01)*randn(nbsensors,latent_dim-nbsources);
    %DeltaB=DeltaB*0.05/norm(DeltaB,'fro')*norm(B,'fro');
    B2=B+DeltaB;
    
    w_function=weight_function(nbiter,nbsamples);
    
    BB2_cell=cell(1,Q);
    AA2_cell=cell(1,Q);
    for q=1:Q
        BB2=(1-w_function).*B(:,q)+w_function.*B2(:,q);
        BB2_cell{q}=BB2;
        AA2=(1-w_function).*A(:,q)+w_function.*A2(:,q);
        AA2_cell{q}=AA2;
    end
    
    V=zeros(nbsensors,nbsamples*nbiter);
    Y=zeros(nbsensors,nbsamples*nbiter);
    for i=0:nbiter-1
        mixtureB=zeros(nbsensors,nbsamples);
        mixtureA=zeros(nbsensors,nbsamples);
        for q=1:Q
            BB2=BB2_cell{q};
            mixtureB=mixtureB+BB2(:,nbsamples*i+1:nbsamples*i+nbsamples).*S(q,:);
            AA2=AA2_cell{q};
            mixtureA=mixtureA+AA2(:,nbsamples*i+1:nbsamples*i+nbsamples).*D(q,:);
        end
        V(:,nbsamples*i+1:nbsamples*i+nbsamples)=mixtureB+noise;
        Y(:,nbsamples*i+1:nbsamples*i+nbsamples)=mixtureA;
    end
    
    Y=Y+V;
        
    %V=B*S+noise;
    %Y=A*D+V;

end