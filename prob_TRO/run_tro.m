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
nbnodes=30;
% Number of channels per node
nbsensors_vec=15*ones(1,nbnodes);
% Number of channels in total
nbsensors=sum(nbsensors_vec);

% Number of samples of the signals
nbsamples=10000;

% Number of filters of X
Q=5;

norm_error=cell(mc_runs,1);

for n_runs=1:mc_runs

    % Create the data
    [Y,V]=create_data(nbsensors_vec,nbnodes,nbsamples);
    
    % Structure related to the data
    Y_cell=cell(2,1);
    Y_cell{1}=Y;
    Y_cell{2}=V;
    Gamma_cell{1}=eye(nbsensors);
    
    data=struct;
    data.Y_cell=Y_cell;
    data.B_cell={};
    data.Gamma_cell=Gamma_cell;
    data.Glob_Const_cell={};

    % Structure related to parameters of the problem
    prob_params=struct;
    prob_params.nbsensors=nbsensors;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensors_vec=nbsensors_vec;
    prob_params.nbsamples=nbsamples;
    
    % Estimate filter using the centralized algorithm
    [X_star,f_star]=tro_solver(data,prob_params);
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
    
    % Solve the TRO problem using TI-DSFO
    [X_est,f_seq,norm_diff,norm_err]=dsfo(data,prob_params,...
        conv,@tro_eval,@tro_solver,@tro_resolve_uniqueness);
    
    norm_error{n_runs}=norm_err;
    
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

function [Y,V]=create_data(nbsensors_vec,nbnodes,nbsamples)

    noisepower=0.1; 
    signalvar=0.5;
    nbsources=5;
    latent_dim=10;

    rng('shuffle');
    d=randn(nbsamples,nbsources);
    d=sqrt(signalvar)./(sqrt(var(d))).*(d-ones(nbsamples,1)*mean(d));
    s=randn(nbsamples,latent_dim-nbsources);
    s=sqrt(signalvar)./(sqrt(var(s))).*(s-ones(nbsamples,1)*mean(s));
    
%     Ainit=rand(nbsources,sum(nbsensors_vec))-0.5;
%     Binit=rand(latent_dim-nbsources,sum(nbsensors_vec))-0.5;
%     noise=randn(nbsamples,sum(nbsensors_vec)); 
%     noise=sqrt(noisepower)./sqrt(var(noise)).*(noise-ones(nbsamples,1)*mean(noise)); 

    for k=1:nbnodes
        Ainit{k}=rand(nbsources,nbsensors_vec(k))-0.5;
        Binit{k}=rand(latent_dim-nbsources,nbsensors_vec(k))-0.5;
        noise{k}=randn(nbsamples,nbsensors_vec(k)); 
        noise{k}=sqrt(noisepower)./sqrt(var(noise{k})).*(noise{k}-ones(nbsamples,1)*mean(noise{k})); 
    end

    column_blk=0;

    Y_cell=cell(nbnodes,1);
    V_cell=cell(nbnodes,1);
    
    for k=1:nbnodes
        V_cell{k}=s*Binit{k}+noise{k};
        Y_cell{k}=d*Ainit{k}+V_cell{k};

        Y(1:nbsamples,column_blk+1:column_blk+nbsensors_vec(k))=Y_cell{k};
        V(1:nbsamples,column_blk+1:column_blk+nbsensors_vec(k))=V_cell{k};
        column_blk=column_blk+nbsensors_vec(k);
    end
    
    %V=s*Binit+noise;
    %Y=d*Ainit+V;
    
    Y=Y';
    V=V';

end

