function X_cell_out=cca_select_sol(X_ref_cell,X_cell,prob_params,data,q)

% Resolve the sign ambiguity for the CCA problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    X=X_cell{1};
    W=X_cell{2};
    Q=prob_params.Q;
    X_ref=X_ref_cell{1};
    W_ref=X_ref_cell{2};
    
    for l=1:Q
        if norm(X_ref(:,l)-X(:,l))>norm(-X_ref(:,l)-X(:,l))
            X(:,l)=-X(:,l);
            W(:,l)=-W(:,l);
        end
    end
    
    X_cell_out=cell(2,1);
    X_cell_out{1}=X;
    X_cell_out{2}=W;
    
end
