% Example script to run the TI-DSFO algorithm to solve the TRO problem

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

addpath('../DSFO/');

% Number of Monte-Carlo runs
mc_runs=5;

% Number of nodes
nbnodes=10;
% Number of channels per node
nbsensors_vec=5*ones(1,nbnodes);
% Number of channels in total
nbsensors=sum(nbsensors_vec);

% Number of samples of the signals
nbsamples=10000;

% Number of filters of X
Q=3;

norm_error=cell(mc_runs,1);
n_runs=1;

while n_runs<=mc_runs
    
    % Create the data
    [Y,B,alpha,c,d]=create_data(nbsensors,nbsamples,Q);
    
    % Structure related to parameters of the problem
    Y_cell{1}=Y;
    B_cell{1}=B;
    B_cell{2}=c;
    Gamma_cell{1}=eye(nbsensors);
    Glob_Const_cell{1}=alpha;
    Glob_Const_cell{2}=d;
    
    data=struct;
    data.Y_cell=Y_cell;
    data.B_cell=B_cell;
    data.Gamma_cell=Gamma_cell;
    data.Glob_Const_cell=Glob_Const_cell;

    % Structure related to parameters of the problem
    prob_params=struct;
    prob_params.nbsensors=nbsensors;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensors_vec=nbsensors_vec;
    prob_params.nbsamples=nbsamples;
    
    % Estimate filter using the centralized algorithm
    [X_star,f_star]=qcqp_solver(data,prob_params);
    prob_params.X_star=X_star;
    % Compute the distance to X^* if equal to 1 
    prob_params.compare_opt=1;
    % Show a dynamic plot if equal to 1
    prob_params.plot_dynamic=0;

    % Structure related to stopping conditions. The last to be
    % achieved stops the algorihtm (To choose only one condition, set the
    % other to a negative value).
    conv=struct;
    conv.tol_f=-1;
    conv.nbiter=200;
    
    % Create adjacency matrix (hollow matrix) of a random graph
    adj=randi(2,nbnodes,nbnodes)-1;
    graph_adj=triu(adj,1)+tril(adj',-1);
    prob_params.graph_adj=graph_adj;
    
    try
        [X_est,f_seq,norm_diff,norm_err]=dsfo(data,prob_params,...
                                conv,@qcqp_eval,@qcqp_solver,[]);
        norm_error{n_runs}=norm_err;
        n_runs=n_runs+1;
    catch
        warning('Infeasible')
    end


end

%%
% Plot the normalized error

x_int=[1:conv.nbiter];
q_5=quantile(cell2mat(norm_error),0.5);
q_25=quantile(cell2mat(norm_error),0.25);
q_75=quantile(cell2mat(norm_error),0.75);
loglog(q_5,'b','LineWidth',2);

hold on
fill([x_int,fliplr(x_int)],[q_5,fliplr(q_75)],'b','FaceAlpha','0.2','LineStyle','none')
fill([x_int,fliplr(x_int)],[q_5,fliplr(q_25)],'b','FaceAlpha','0.2','LineStyle','none')
xlim([1,inf])
ylim([1e-6,inf])

ax=gca;
ax.FontSize=14;
xlabel('Iterations','Interpreter','latex','Fontsize',20)
ylabel('$\frac{||X^{i}-X^*||_2^F}{||X^*||_F^2}$','Interpreter','latex','Fontsize',20)
grid on

%%

function [Y,B,alpha,c,d]=create_data(nbsensors,nbsamples,Q)

    rng('shuffle');
    Y=randn(nbsensors,nbsamples);
    Y=Y-mean(Y,2);
    Y=Y./var(Y')';
    Ryy=1/nbsamples*conj(Y*Y');
    Ryy=make_sym(Ryy);
    B=randn(nbsensors,Q);
    c=randn(nbsensors,1);
    d=randn(Q,1);
    w=(B'*inv(Ryy)'*c-d)./(c'*inv(Ryy)*c);
    X=inv(Ryy)*(B-c*w');
    
    toss=randi(2)-1;
    
    if toss==0
        alpha=randn(1,1);
        alpha=alpha^2;
    % Enforce the solution to be strictly satisfying the inequality
    % constraint
    else
        alpha=randn(1,1);
        alpha=alpha^2;
        alpha=sqrt(norm(X,'fro')^2+alpha^2);
    end

    % If problem is infeasible, recreate the parameters
    while(alpha^2<norm(d)^2/norm(c)^2)
        c=randn(nbsensors,1);
        d=randn(1,1);
        w=(B'*inv(Ryy)'*c-d)/(c'*inv(Ryy)*c);
        X=inv(Ryy)*(B-c*w');
        toss=randi(2)-1;
        if toss==0
            alpha=randn(1,1);
            alpha=alpha^2;
        else
            alpha=randn(1,1);
            alpha=alpha^2;
            alpha=sqrt(norm(X,'fro')^2+alpha^2);
        end
    end

end

