% Example script to run the TI-DSFO algorithm to solve the TRO problem

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

addpath('../dsfo/DTRO_DACGEE/')

% Number of Monte-Carlo runs
mc_runs=5;

% Number of nodes
nbnodes=30;
% Number of channels per node
nbsensors_vec=15*ones(1,nbnodes);
% Number of channels in total
nbsensors=sum(nbsensors_vec);

% Number of samples of the signals
nbsamples=10000;

% Number of filters of X
Q=5;

Q_data=5;
% Total number of point sources
D=10;

norm_error=cell(mc_runs,1);

for n_runs=1:mc_runs

    % Create the data
    [Y,V]=create_data(nbsamples,nbsensors_vec,nbnodes,Q_data,D);
    Y=Y';
    V=V';

    % Structure related to parameters of the problem
    prob_params=struct;
    prob_params.nbsensors=nbsensors;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensors_vec=nbsensors_vec;
    % Compute the distance to X^* if equal to 1 
    prob_params.compare_opt=1;
    % Show a dynamic plot if equal to 1
    prob_params.plot_dynamic=0;

    % Structure related to the data
    Y_cell=cell(2,1);
    Y_cell{1}=Y;
    Y_cell{2}=V;
    Gamma_cell{1}=eye(nbsensors);
    B_cell={};
    
    data=struct;
    data.Y_cell=Y_cell;
    data.B_cell=B_cell;
    data.Gamma_cell=Gamma_cell;
    
    % Estimate filter using the centralized algorithm
    [X_star,f_star]=tro_solver(prob_params,data);


    % Structure related to convergence conditions. The first to be
    % achieved stops the algorihtm (To choose only one condition, set the
    % other to a negative value).
    conv=struct;
    conv.tol_f=-1;
    conv.nbiter=200;
    
    % Create adjacency matrix (hollow matrix) of a random graph
    adj=randi(2,nbnodes,nbnodes)-1;
    graph_adj=triu(adj,1)+tril(adj',-1);
    prob_params.graph_adj=graph_adj;
    
    % Solve the TRO problem using TI-DSFO
    [~,f_track,~,norm_star_track]=ti_dsfo(prob_params,data,@tro_eval,...
        @tro_solver,@tro_resolve_uniqueness,conv,X_star);
    
    
    norm_error{n_runs}=norm_star_track;
    
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

xlabel('Iterations','Interpreter','latex')
ylabel('$\frac{1}{||X^*||_F^2}||X^{i}-X^*||_2^F$','Interpreter','latex')
grid on



