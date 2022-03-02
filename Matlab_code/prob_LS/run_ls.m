% Example script to run the DSFO algorithm to solve the Least Squares (LS)
% Problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

addpath('../dsfo_toolbox/');

% Number of Monte-Carlo runs.
mc_runs=5;

% Number of nodes.
nbnodes=10;
% Number of channels per node.
nbsensors_vec=5*ones(nbnodes,1);
% Number of channels in total.
nbsensors=sum(nbsensors_vec);

% Number of samples of the signals.
nbsamples=10000;

% Number of filters of X.
Q=3;

norm_error=cell(mc_runs,1);
n_runs=1;

while n_runs<=mc_runs
    
    % Create the data.
    [Y,D]=create_data(nbsensors_vec,nbnodes,nbsamples,Q);
    
    % Structure related to the data of the problem.
    Y_cell{1}=Y;
    B_cell={};
    Gamma_cell={};
    Glob_Const_cell{1}=D;
    
    data=struct;
    data.Y_cell=Y_cell;
    data.B_cell=B_cell;
    data.Gamma_cell=Gamma_cell;
    data.Glob_Const_cell=Glob_Const_cell;

    % Structure related to parameters of the problem.
    prob_params=struct;
    prob_params.nbsensors=nbsensors;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensors_vec=nbsensors_vec;
    prob_params.nbsamples=nbsamples;
    
    % Random updating order.
    prob_params.update_path=randperm(nbnodes);
    
    % Estimate filter using the centralized algorithm.
    [X_star,f_star]=ls_solver(prob_params,data);
    prob_params.X_star=X_star;
    % Compute the distance to X_star if "true".
    prob_params.compare_opt=true;
    % Show a dynamic plot if "true".
    prob_params.plot_dynamic=false;

    % Structure related to stopping conditions. We fix the number of 
    % iterations the DSFO algorithm will perform to 200.
    conv=struct;
    conv.nbiter=200;
    
    % Create adjacency matrix (hollow matrix) of a random graph.
    adj=randi(2,nbnodes,nbnodes)-1;
    graph_adj=triu(adj,1)+tril(adj',-1);
    non_connected=find(all(graph_adj==0));
    % If there are non-connected nodes, connect them to every other node.
    if(non_connected)
        graph_adj(:,non_connected)=1;
        graph_adj(non_connected,:)=1;
    end
    prob_params.graph_adj=graph_adj;
    
    % Solve the LS in a distributed way using the DSFO framework.
    [X_est,norm_diff,norm_err,f_seq]=dsfo(prob_params,data,...
                @ls_solver,conv,[],@ls_eval);
    norm_error{n_runs}=norm_err;
    n_runs=n_runs+1;
    


end

%%
% Plot the normalized error.

x_int=[1:conv.nbiter];
q_5=quantile(cell2mat(norm_error),0.5);
q_25=quantile(cell2mat(norm_error),0.25);
q_75=quantile(cell2mat(norm_error),0.75);
loglog(q_5,'b','LineWidth',2);

hold on
fill([x_int,fliplr(x_int)],[q_5,fliplr(q_75)],'b','FaceAlpha','0.2','LineStyle','none')
fill([x_int,fliplr(x_int)],[q_5,fliplr(q_25)],'b','FaceAlpha','0.2','LineStyle','none')
xlim([1,inf])
ylim([1e-20,inf])

ax=gca;
ax.FontSize=14;
xlabel('Iterations','Interpreter','latex','Fontsize',20)
ylabel('$\epsilon$','Interpreter','latex','Fontsize',20)
grid on

%%

function [Y,D]=create_data(nbsensors_vec,nbnodes,nbsamples,Q)

    signalvar=0.5;
    noisepower=0.1;
    nbsources=Q;
    offset=0.5;
    
    rng('shuffle');
    D=randn(nbsources,nbsamples);
    D=sqrt(signalvar)./(sqrt(var(D,0,2))).*(D-mean(D,2)*ones(1,nbsamples));

    A=cell(nbnodes,1);
    for k=1:nbnodes
        A{k}=rand(nbsensors_vec(k),nbsources)-offset;
        noise{k}=randn(nbsensors_vec(k),nbsamples); 
        noise{k}=sqrt(noisepower)./sqrt(var(noise{k},0,2)).*(noise{k}...
            -mean(noise{k},2)*ones(1,nbsamples));
    end

    column_blk=0;

    for k=1:nbnodes
        Y_cell{k}=A{k}*D+noise{k};
        Y(column_blk+1:column_blk+nbsensors_vec(k),1:nbsamples)=Y_cell{k};
        column_blk=column_blk+nbsensors_vec(k);
    end

end

