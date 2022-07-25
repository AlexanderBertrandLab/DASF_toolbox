function X_cell_out=cca_select_sol(X_ref_cell,X_cell,nbsensors_vec,q)

% Resolve the sign ambiguity for the CCA problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    X=X_cell{1};
    W=X_cell{2};
    M_X=size(X,1);
    Q=size(X,2);
    X_ref=X_ref_cell{1};
    W_ref=X_ref_cell{2};
    M=sum(nbsensors_vec);
    
    if(M_X==M)
    % Comparison of the full variable (resolve the sign ambiguity between X
    % and X_star).
        for l=1:Q
            if norm(X_ref(:,l)-X(:,l))>norm(-X_ref(:,l)-X(:,l))
                X(:,l)=-X(:,l);
                W(:,l)=-W(:,l);
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
                W(:,l)=-W(:,l);
            end
        end
    end
    
    X_cell_out=cell(2,1);
    X_cell_out{1}=X;
    X_cell_out{2}=W;
    
end
