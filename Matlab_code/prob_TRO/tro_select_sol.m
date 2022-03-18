function X=tro_select_sol(X_ref,X,nbsensors_vec,q)

% Resolve the sign ambiguity for the TRO problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    M_X=size(X_ref,1);
    Q=size(X_ref,2);
    M=sum(nbsensors_vec);
    
    if(M_X==M)
    % Comparison of the full variable (resolve the sign ambiguity between X
    % and X_star).
        for l=1:Q
            if norm(X_ref(:,l)-X(:,l))>norm(-X_ref(:,l)-X(:,l))
                X(:,l)=-X(:,l);
            end
        end
    else
    % Comparison of the blocks (resolve the sign ambiguity between X and
    % X_old).
        Xq=X(1:nbsensors_vec(q),:);
        Xq_ref=X_ref(1:nbsensors_vec(q),:);
        for l=1:Q
            if norm(Xq_ref(:,l)-Xq(:,l))>norm(-Xq_ref(:,l)-Xq(:,l))
                X(:,l)=-X(:,l);
            end
        end
    end

end