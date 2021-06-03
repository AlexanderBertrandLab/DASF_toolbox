function X=tro_resolve_uniqueness(X_old,X)

% Resolve the sign ambiguity for the TRO problem

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Q=size(X_old,2);

    for l=1:Q
        if sum(sum((X_old(:,l)-X(:,l)).^2))>sum(sum((-X_old(:,l)-X(:,l)).^2))
            X(:,l)=-X(:,l);
        end
    end

end