function X=tro_select_sol(X_ref,X)

% Resolve the sign ambiguity for the TRO problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Q=size(X_ref,2);

    for l=1:Q
        if norm(X_ref(:,l)-X(:,l))>norm(-X_ref(:,l)-X(:,l))
            X(:,l)=-X(:,l);
        end
    end

end