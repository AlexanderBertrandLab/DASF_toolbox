function dynamic_plot(X,X_star)

% Plot the first column of X and X_star.
%
% INPUTS:
% X (nbsensors x Q): Global variable equal.
% X_star (nbsensors x Q): Optimal solution.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

    plot(X_star(:,1),'r')
    hold on
    plot(X(:,1),'b')
    ylim([1.2*min(real(X_star(:,1))) 1.2*max(real(X_star(:,1)))]);
    hold off
    drawnow
    
end