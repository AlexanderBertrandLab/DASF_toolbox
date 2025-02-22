% Example script to run the F-DASF algorithm on the TRO problem

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

Q=2;

% Number of Monte-Carlo runs
mc_runs=10;

data_cell_fdasf=cell(mc_runs,1);
data_cell_dasf=cell(mc_runs,1);
data_cell_iter=cell(mc_runs,1);

h=waitbar(0,'Computing');

for n_runs=1:mc_runs
    
    [Y,V]=create_data(nbsensors,nbsamples);

    Ryy=1/nbsamples*conj(Y*Y');
    Ryy=make_sym(Ryy);
    Rvv=1/nbsamples*conj(V*V');
    Rvv=make_sym(Rvv);
    data=struct;
    data.Ryy=Ryy;
    data.Rvv=Rvv;

    params=struct;
    params.nbsensors=nbsensors;
    params.Q=Q;
    params.nbnodes=nbnodes;
    params.nbsensors_vec=nbsensors_vec;

    debug=0;

    % Centralized trace ratio, used as ground truth
    [X_star,rho_star]=tro_centralized(Ryy,Rvv,eye(nbsensors),Q);

    conv=struct;
    conv.tol_rho=-1;
    conv.nbiter=500;
    
    p=struct;
    p.connected=1;
    G=gsp_erdos_renyi(params.nbnodes,0.8,p);
    er_adj=double(full(G.A));
    params.adj=er_adj;

    [X_fdasf,rho_track_fdasf,error_track_fdasf]=...
        tro_fdasf(params,data,conv,debug,X_star);
    [X_dasf,rho_track_dasf,error_track_dasf,iter_count_vec]=...
        tro_dasf(params,data,conv,debug,X_star);
    
    data_cell_fdasf{n_runs}=error_track_fdasf;
    data_cell_dasf{n_runs}=error_track_dasf;
    data_cell_iter{n_runs}=iter_count_vec;
    
    waitbar(n_runs/mc_runs,h,[sprintf('%3.2f',n_runs/mc_runs*100),'%'])

end

close(h)


%%
% Plot the MSE

set(groot,'defaultAxesTickLabelInterpreter','latex');
conv.nbiter=500;

x_int=[1:conv.nbiter];
spacing=logspace(0,log10(conv.nbiter),20);
spacing=round(spacing);
q_5=quantile(cell2mat(data_cell_fdasf),0.5);
loglog(x_int,q_5,'b','LineWidth',1.5)
hold on
p1=loglog(spacing,q_5(spacing),'b.','Marker','d','MarkerSize',10,'LineWidth',1,'DisplayName','F-DASF');

x_int=[1:conv.nbiter];
q_5=quantile(cell2mat(data_cell_dasf),0.5);
loglog(x_int,q_5,'b','LineWidth',1.5)
p3=loglog(spacing,q_5(spacing),'b.','Marker','.','MarkerSize',15,'LineWidth',1,'DisplayName','DASF');

hold on
ylim([1e-9,10^1])
ax=gca;
ax.YAxis(1).FontSize=10;
ylabel('MedSE $\epsilon$','Interpreter','latex','FontSize',20)

yyaxis right
ylim([1,5e3])
ylabel({'Number of'; 'auxiliary problems'},'Interpreter','latex','FontSize',20)

x_int=[1:conv.nbiter];
q_5=quantile(cell2mat(data_cell_fdasf),0.5);
loglog(x_int,x_int,'r','LineWidth',1.5)
p2=loglog(x_int(spacing),x_int(spacing),'rd','MarkerSize',10,'LineWidth',0.5,'DisplayName','F-DASF');

ind_q5=quantile(cell2mat(data_cell_iter),0.5);
inds=cumsum(ind_q5);
avg_inds=mean(ind_q5);

q_5=quantile(cell2mat(data_cell_dasf),0.5);
loglog(x_int,inds,'r-','LineWidth',1.5)
p4=loglog(x_int(spacing),inds(spacing),'r.','MarkerSize',15,'LineWidth',0.5,'DisplayName','DASF');

rax = findall(0, 'YAxisLocation','right');
set(rax,'Yscale','log','box','off');

ax=gca;
ax.YAxis(1).Color = 'b';
ax.YAxis(2).Color = 'r';
set(gca,'FontSize',18)
xlim([1,conv.nbiter])
xlabel('Iteration index $i$','Interpreter','latex')
grid on

p5=loglog(rand,'k-','LineWidth',2,'Visible','off');
p6=loglog(rand,'k*','LineWidth',1,'Visible','off');
p7=fill([1 2],[1 2],'b','LineStyle','-','LineWidth',2,'Visible','off');
p8=fill([1 2],[1 2],'r','LineStyle','-','LineWidth',2,'Visible','off');

p9=loglog(rand,'kd','MarkerSize',10,'LineWidth',1,'Visible','off');
p10=loglog(rand,'k.','MarkerSize',15,'LineWidth',1,'Visible','off');

[hl(1).leg, hl(1).obj, hl(1).hout, hl(1).mout] = ...
legendflex([p9;p10], {'F-DASF','DASF'}, ...
'anchor', {'sw','sw'}, ...
'ncol',1,...
'buffer', [7 150], ...
'fontsize',16, ...
'title', '\textbf{Algorithm}',...
'interpreter','latex');

set(gcf,'Position',[440   378   660   360])



