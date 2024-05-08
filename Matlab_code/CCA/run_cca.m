% Example script to run the DASF algorithm to solve the CCA Problem.

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

addpath('../dasf_toolbox/');

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
Q=2;

% Number of variables.
nbvariables=2;

norm_error=cell(mc_runs,1);

for n_runs=1:mc_runs

    % Create the data.
    [Y,V]=create_data(nbsensors_vec,nbnodes,nbsamples);
    
    % Structure related to the data of the problem.
    Y_cell{1}=Y;
    V_cell{1}=V;
    
    data_X=struct;
    data_X.Y_cell=Y_cell;
    data_X.B_cell={};
    data_X.Gamma_cell={};
    data_X.Glob_Const_cell={};
    
    data_W=struct;
    data_W.Y_cell=V_cell;
    data_W.B_cell={};
    data_W.Gamma_cell={};
    data_W.Glob_Const_cell={};
    
    data=cell(nbvariables,1);
    data{1}=data_X;
    data{2}=data_W;

    % Structure related to parameters of the problem.
    prob_params=struct;
    prob_params.nbsensors=nbsensors;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensors_vec=nbsensors_vec;
    prob_params.nbsamples=nbsamples;
    prob_params.nbvariables=nbvariables;
    
    % Random updating order.
    prob_params.update_path=randperm(nbnodes);
    
    % Estimate filter using the centralized algorithm.
    [X_star,f_star]=cca_solver(prob_params,data);
    prob_params.X_star=X_star;
    % Compute the distance to X_star if "true".
    prob_params.compare_opt=true;
    % Show a dynamic plot if "true".
    prob_params.plot_dynamic=false;

    % Structure related to stopping conditions. We fix the number of 
    % iterations the DASF algorithm will perform to 1000.
    conv=struct;
    conv.nbiter=1000;
    
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
    
    % Solve the CCA problem in a distributed way using the DASF framework.
    [X_est,norm_diff,norm_err,f_seq]=dasf_multivar(prob_params,data,...
                    @cca_solver,conv,@cca_select_sol,@cca_eval);
    
    norm_error{n_runs}=norm_err;
    
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
ylim([1e-8,inf])

ax=gca;
ax.FontSize=14;
xlabel('Iterations','Interpreter','latex','Fontsize',20)
ylabel('$\epsilon$','Interpreter','latex','Fontsize',20)
grid on

%%

function [Y,V]=create_data(nbsensors_vec,nbnodes,nbsamples)
    
    noisepower=0.1;
    signalvar=0.5;
    nbsources=10;
    offset=0.5;
    lags=3;

    rng('shuffle');
    D=randn(nbsources,nbsamples+lags);
    D=sqrt(signalvar)./(sqrt(var(D,0,2))).*(D-mean(D,2)*ones(1,nbsamples+lags));

    for k=1:nbnodes
        A{k}=rand(nbsensors_vec(k),nbsources)-offset;
        noise{k}=randn(nbsensors_vec(k),nbsamples+lags); 
        noise{k}=sqrt(noisepower)./sqrt(var(noise{k},0,2)).*(noise{k}...
            -mean(noise{k},2)*ones(1,nbsamples+lags)); 
    end

    column_blk=0;

    signal_cell=cell(nbnodes,1);
    Y_cell=cell(nbnodes,1);
    V_cell=cell(nbnodes,1);
    
    for k=1:nbnodes
        signal_cell{k}=A{k}*D+noise{k};
        Y_cell{k}=signal_cell{k}(:,1:nbsamples);
        V_cell{k}=signal_cell{k}(:,lags+1:end);

        Y(column_blk+1:column_blk+nbsensors_vec(k),1:nbsamples)=Y_cell{k};
        V(column_blk+1:column_blk+nbsensors_vec(k),1:nbsamples)=V_cell{k};
        column_blk=column_blk+nbsensors_vec(k);
    end

end

