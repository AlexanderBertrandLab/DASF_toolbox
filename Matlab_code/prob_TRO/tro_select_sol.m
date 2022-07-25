function X=tro_select_sol(X_ref,X,prob_params,data,q)

% Resolve the sign ambiguity for the TRO problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    Q=prob_params.Q;
    
    for l=1:Q
        if norm(X_ref(:,l)-X(:,l))>norm(-X_ref(:,l)-X(:,l))
            X(:,l)=-X(:,l);
        end
    end

end