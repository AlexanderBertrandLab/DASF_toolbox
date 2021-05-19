% Example script to run the TI-DTRO algorithm.
clear all
close all

addpath('../DTRO_DACGEE/');

mc_runs=20;

nbnodes=30;
nbsensnode=15*ones(1,nbnodes);
nbsamples=10000;

nbsens=sum(nbsensnode);

Q_data=5;
Q=5;
% Total number of point sources
D=10;

for n_runs=1:mc_runs

    [Y,V]=create_data(nbsamples,nbsensnode,nbnodes,Q_data,D);
    Y=Y';
    V=V';
    Y_cell=cell(2,1);
    Y_cell{1}=Y;
    Y_cell{2}=V;
    Gamma_cell{1}=eye(nbsens);
    B_cell={};

    prob_params=struct;
    prob_params.nbsens=nbsens;
    prob_params.Q=Q;
    prob_params.nbnodes=nbnodes;
    prob_params.nbsensnode=nbsensnode;
    prob_params.denom_sum=0;
    prob_params.sgn_sync=1;

    conv=struct;
    conv.tol_rho=1e-12;
    conv.nbiter=-1;

    debug=1;

    % Estimate filter

    data=struct;
    data.Y_cell=Y_cell;
    data.B_cell=B_cell;
    data.Gamma_cell=Gamma_cell;
    [X_star,f_star]=tro_solver(prob_params,data);

    % Distributed trace ratio

    conv.tol_f=-1;
    conv.nbiter=200;
    prob_params.efficient=0;
    
    p=struct;
    p.connected=1;
    G=gsp_erdos_renyi(prob_params.nbnodes,0.6,p);
    er_adj=double(full(G.A));
    er_graph=graph(er_adj);
    bins=conncomp(er_graph);
    prob_params.graph_adj=er_adj;
    [~,f_track,~,norm_star_track]=ti_dsfo(prob_params,data,@tro_eval,@tro_solver,conv,debug,X_star);
    
    prob_params.denom_sum=0;
    clear data
    data=struct;
    Ryy=1/10000*conj(Y*Y');
    Rvv=1/10000*conj(V*V');
    Ryy=make_sym(Ryy);
    Rvv=make_sym(Rvv);
    data.R_first=Ryy;
    data.R_second=Rvv;
    conv.tol_rho=-1;
    [~,f_track_2,~,norm_star_track_2]=ti_dtro_alt(prob_params,data,er_adj,conv,debug,X_star);

    msec{n_runs}=norm_star_track;
    
end




