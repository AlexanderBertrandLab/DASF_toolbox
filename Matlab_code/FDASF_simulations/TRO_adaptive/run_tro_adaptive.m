% Example script to run the F-DASF algorithm on the TRO problem in an
% adaptive setting

% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be

clear all
close all

nbnodes=10;
nbsensors_vec=5*ones(1,nbnodes);
nbsamples=1000;
nbsensors=sum(nbsensors_vec);
Q=2;

% Number of Monte-Carlo runs
mc_runs=2;

data_cell_fdasf=cell(mc_runs,1);
data_cell_dasf=cell(mc_runs,1);

h=waitbar(0,'Computing');

for n_runs=1:mc_runs

    params=struct;
    params.nbsensors=nbsensors;
    params.Q=Q;
    params.nbnodes=nbnodes;
    params.nbsensors_vec=nbsensors_vec;
    params.nbsamples=nbsamples;
    
    debug=0;
    
    conv=struct;
    conv.tol_rho=-1;
    conv.nbiter=500;
    
    [Y_full,V_full]=create_data(nbsensors,nbsamples,Q,conv.nbiter);
    data=struct;
    data.Y_full=Y_full;
    data.V_full=V_full;
    
    p=struct;
    p.connected=1;
    G=gsp_erdos_renyi(params.nbnodes,0.8,p);
    er_adj=double(full(G.A));
    params.adj=er_adj;

    [X_fdasf,rho_track_fdasf,error_track_fdasf]=...
        tro_fdasf(params,data,conv,debug);
    
    [X_dasf,rho_track_dasf,error_track_dasf]=...
        tro_dasf(params,data,conv,debug);
    
    data_cell_fdasf{n_runs}=error_track_fdasf;
    data_cell_dasf{n_runs}=error_track_dasf;
    
    waitbar(n_runs/mc_runs,h,[sprintf('%3.2f',n_runs/mc_runs*100),'%'])

end

close(h)

%%
% Plot the MSE

nbiter=500;
nbnodes=10;
nbsamples=1000;

spacing=linspace(1,nbiter*nbsamples,20);
spacing=round(spacing);

set(groot,'defaultAxesTickLabelInterpreter','latex');

hold on
q_5=repelem(quantile(cell2mat(data_cell_fdasf),0.5),nbsamples);
semilogy(q_5,'Color','b','LineWidth',1);
semilogy(spacing,q_5(spacing),'b.','Marker','d','MarkerSize',10,'LineWidth',2,'DisplayName','F-DASF');

q_5=repelem(quantile(cell2mat(data_cell_dasf),0.5),nbsamples);
semilogy(q_5,'Color','b','LineWidth',1);
semilogy(spacing,q_5(spacing),'b.','Marker','.','MarkerSize',15,'LineWidth',2,'DisplayName','DASF');

ylim([1e-6,10^1])
ax=gca;
ax.YAxis(1).FontSize=10;
ylabel('MedSE $\epsilon$','Interpreter','latex','FontSize',20)
rax = findall(0, 'YAxisLocation','left');
set(rax,'YScale','log','box','off');

yyaxis right
ylabel('$p(t)$','Interpreter','latex','FontSize',20)
ylim([-0.1,1.1])

w_function=weight_function(nbiter,nbsamples);
plot(w_function,'r','LineWidth',2);

rax = findall(0, 'YAxisLocation','right');
set(rax,'box','off');

ax=gca;
ax.YAxis(1).Color = 'b';
ax.YAxis(2).Color = 'r';
set(gca,'FontSize',18)
xlim([1,nbiter*nbsamples])
xlabel('Time $t$','Interpreter','latex')
grid on

p1=loglog(rand,'kd','MarkerSize',10,'LineWidth',1,'Visible','off');
p2=loglog(rand,'k.','MarkerSize',15,'LineWidth',1,'Visible','off');

[hl(1).leg, hl(1).obj, hl(1).hout, hl(1).mout] = ...
legendflex([p1;p2], {'F-DASF','DASF'}, ...
'anchor', {'sw','sw'}, ...
'ncol',1,...
'buffer', [15 200], ...
'fontsize',16, ...
'title', '\textbf{Algorithm}',...
'interpreter','latex');

set(gcf,'Position',[440   378   660   360])



