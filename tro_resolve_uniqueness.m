function X=tro_resolve_uniqueness(X_old,X)

    Q=size(X_old,2);

    for l=1:Q
        if sum(sum((X_old(:,l)-X(:,l)).^2))>sum(sum((-X_old(:,l)-X(:,l)).^2))
            X(:,l)=-X(:,l);
        end
    end

end