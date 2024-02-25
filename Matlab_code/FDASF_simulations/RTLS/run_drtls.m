% Example script to run the F-DASF algorithm on the RTLS problem

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

nbnodes=10;
nbsensors_vec=5*ones(1,nbnodes);
nbsamples=10000;
nbsensors=sum(nbsensors_vec); 
Q=1;

% Number of Monte-Carlo runs
mc_runs=10;

data_cell=cell(mc_runs,1);
data_cell_fdasf=cell(mc_runs,1);
index_cell=cell(mc_runs,1);

h=waitbar(0,'Computing');

for n_runs=1:mc_runs
    
    [Y,D]=create_data(nbsensors_vec,nbnodes,nbsamples,Q);
    L_vec=sqrt(0.1)*randn(nbsensors,1);
    L_vec=L_vec+1;
    L=diag(L_vec);
    LL=make_sym(L'*L);
    delta=1;
    
    Ryy=1/nbsamples*conj(Y*Y');
    ryd=1/nbsamples*conj(Y*D');
    rdd=1/nbsamples*conj(D*D');
    Ryy=make_sym(Ryy);

    data=struct;
    data.Ryy=Ryy;
    data.ryd=ryd;
    data.rdd=rdd;
    data.LL=LL;
    data.delta=delta;

    params=struct;
    params.nbsens=nbsensors;
    params.Q=Q;
    params.nbnodes=nbnodes;
    params.nbsensnode=nbsensors_vec;

    debug=0;

    % Centralized RTLS, used as ground truth:
    % - Using Dinkelbach's procedure
    [X_star,rho_star]=rtls_solver(Ryy,ryd,rdd,LL,eye(nbsensors),delta);
    % - Without using Dinkelbach's procedure
    [X_star_2,rho_star_2]=rtls_solver_2(Ryy,ryd,rdd,LL,eye(nbsensors),delta);
    
    % Convergence criteria for the distributed RTLS
    conv=struct;
    conv.tol_rho=-1;
    conv.nbiter=50;
    
    p=struct;
    p.connected=1;
    G=gsp_erdos_renyi(params.nbnodes,0.8,p);
    er_adj=double(full(G.A));
    params.adj=er_adj;

    [X,rho_track,norm_track,error_track,ind_full]=dasf_rtls(params,data,conv,debug,X_star);
    [X_fdasf,rho_track_fdasf,norm_track_fdasf,error_track_fdasf]=fdasf_rtls(params,data,conv,debug,X_star);
    
    data_cell{n_runs}=error_track;
    data_cell_fdasf{n_runs}=error_track_fdasf;
    index_cell{n_runs}=ind_full;
    
    waitbar(n_runs/mc_runs,h,[sprintf('%3.2f',100*n_runs/mc_runs),'%'])

end

close(h)

%%
% Plot the MSE

set(groot,'defaultAxesTickLabelInterpreter','latex');

x_int=[1:50];
spacing=logspace(0,log10(50),20);
spacing=round(spacing);
q_5=quantile(cell2mat(data_cell_fdasf),0.5);
loglog(x_int,q_5(x_int),'b','LineWidth',1.5)
hold on
p1=loglog(spacing,q_5(spacing),'b.','Marker','d','MarkerSize',10,'LineWidth',1,'DisplayName','F-DASF');

q_5=quantile(cell2mat(data_cell),0.5);
loglog(x_int,q_5(x_int),'b','LineWidth',1.5)
p3=loglog(spacing,q_5(spacing),'b.','Marker','.','MarkerSize',15,'LineWidth',1,'DisplayName','DASF');

hold on
ylim([1e-9,10^1])
ax=gca;
ax.YAxis(1).FontSize=10;
ylabel('MedSE $\epsilon$','Interpreter','latex','FontSize',20)

yyaxis right
ylim([1,1e3])
ylabel({'Number of'; 'auxiliary problems'},'Interpreter','latex','FontSize',20)

q_5=quantile(cell2mat(data_cell_fdasf),0.5);
loglog(x_int,x_int,'r','LineWidth',1.5)
p2=loglog(x_int(spacing),x_int(spacing),'rd','MarkerSize',10,'LineWidth',0.5,'DisplayName','F-DASF');

ind_q5=quantile(cell2mat(index_cell'),0.5);
inds=cumsum(ind_q5);
avg_inds=mean(ind_q5);

q_5=quantile(cell2mat(data_cell),0.5);
loglog(x_int,inds(x_int),'r-','LineWidth',1.5)
p4=loglog(x_int(spacing),inds(spacing),'r.','MarkerSize',15,'LineWidth',0.5,'DisplayName','DASF');

rax = findall(0, 'YAxisLocation','right');
set(rax,'Yscale','log','box','off');

ax=gca;
ax.YAxis(1).Color = 'b';
ax.YAxis(2).Color = 'r';
set(gca,'FontSize',18)
xlim([1,50])
xlabel('Iteration index $i$','Interpreter','latex')
grid on

p5=loglog(rand,'k-','LineWidth',2,'Visible','off');
p6=loglog(rand,'k*','LineWidth',1,'Visible','off');
p7=fill([1 2],[1 2],'b','LineStyle','-','LineWidth',2,'Visible','off');
p8=fill([1 2],[1 2],'r','LineStyle','-','LineWidth',2,'Visible','off');

p9=loglog(rand,'kd','LineWidth',1,'Visible','off');
p10=loglog(rand,'k.','MarkerSize',15,'LineWidth',1,'Visible','off');

[hl(1).leg, hl(1).obj, hl(1).hout, hl(1).mout] = ...
legendflex([p9;p10], {'F-DASF','DASF'}, ...
'anchor', {'sw','sw'}, ...
'ncol',1,...
'buffer', [7 100], ...
'fontsize',16, ...
'title', '\textbf{Algorithm}',...
'interpreter','latex');

set(gcf,'Position', [440   378   660   360])


function [Y,D]=create_data(nbsensors_vec,nbnodes,nbsamples,Q)

    signalvar=0.5;
    Avar=0.3;
    noisepower=0.2;
    nbsources=Q;
    
    rng('shuffle');
    D=randn(nbsources,nbsamples);
    D=sqrt(signalvar)./(sqrt(var(D,0,2))).*(D-mean(D,2)*ones(1,nbsamples));

    A=cell(nbnodes,1);
    for k=1:nbnodes
        A{k}=sqrt(Avar)*randn(nbsensors_vec(k),nbsources);
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
    
    D_noise=randn(nbsources,nbsamples);
    D_noisepower=0.02;
    D_noise=sqrt(D_noisepower)./(sqrt(var(D_noise,0,2)))...
        .*(D_noise-mean(D_noise,2)*ones(1,nbsamples));
    D=D+D_noise;

end


