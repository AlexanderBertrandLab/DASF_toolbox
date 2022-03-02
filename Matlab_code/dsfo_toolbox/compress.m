function data_compressed=compress(data,Cq)

% Function to compress the data.
%
% INPUTS:
% data: Structure related to the data.
% Cq: Transformation matrix making the transition between local and global
%     data.
%
% OUTPUTS:
% data_compressed: Structure containing the compressed data. Contains the
%                  same fields as 'data'.
%
% Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
% (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
% Analytics
% Correspondence: cemates.musluoglu@esat.kuleuven.be
    
    Y_cell=data.Y_cell;
    B_cell=data.B_cell;
    Gamma_cell=data.Gamma_cell;
    Glob_Const_cell=data.Glob_Const_cell;
    
    data_compressed=struct;

    if(~isempty(data.Y_cell))
        nbY=length(Y_cell);
        Y_cell_compressed=cell(nbY,1);
        for ind=1:nbY
            Y_cell_compressed{ind}=Cq'*Y_cell{ind};
        end
        data_compressed.Y_cell=Y_cell_compressed;
    else
        data_compressed.Y_cell={};
    end
    
    if(~isempty(data.B_cell))
        nbB=length(B_cell);
        B_cell_compressed=cell(nbB,1);
        for ind=1:nbB
            B_cell_compressed{ind}=Cq'*B_cell{ind};
        end
        data_compressed.B_cell=B_cell_compressed;
    else
        data_compressed.B_cell={};
    end
    
    if(~isempty(data.Gamma_cell))
        nbGamma=length(Gamma_cell);
        Gamma_cell_compressed=cell(nbGamma,1);
        for ind=1:nbGamma
            Gamma_cell_compressed{ind}=Cq'*Gamma_cell{ind}*Cq;
            Gamma_cell_compressed{ind}=make_sym(Gamma_cell_compressed{ind});
        end
        data_compressed.Gamma_cell=Gamma_cell_compressed;
    else
        data_compressed.Gamma_cell={};
    end
    
    if(~isempty(data.Glob_Const_cell))
        data_compressed.Glob_Const_cell=Glob_Const_cell;
    else
        data_compressed.Glob_Const_cell={};
    end
    
end