function [x0,Sys,spgrid,BCtype] = ConstrHeat1D_CDC22(cfun,bd_dist,x0fun,N)
% Finite Differences approximation of a 1D Heat equation with different
% types of distributed or boundary control and observation.
%
% Case 6: Neumann boundary control and disturbance input d_1(t) at x=0, 
% regulated output y(t) at x=1, a Neumann boundary disturbance d_2(t) at 
% x=1. Additional distributed disturbance input d_0(t) with spatial profile
% 'bd_dist'.
%
% Usage: Can also use solely for defining the initial state x0
% Inputs:
%   cfun : [1x1 function_handle] spatially varying
%   thermal diffusivity of the material
% 
%   bd_dist : distributed disturbance input profile
%
%   x0fun : [1x1 function_handle] initial heat profile
%
%   N : [integer] Dimension of the approximated system
%
% Outputs:
%   x0 : [Nx1 matrix] initial state of the plant
%
%   Sys : [struct with fields (at least) A,B,C,D] approximation of the
%   system to be controlled
%
%   spgrid : [1xN matrix] spatial grid of the case
%
%   BCtype : [string] Boundary control type for the case, one of 'NN',
%   'ND', 'DN' or 'DD'.

spgrid = linspace(0,1,N);
h = 1/(N-1);
% Case 1 has Neumann boundary conditions at both x=0 and x=1
BCtype = 'NN';

% modify A and the spatial grid in accordance with the thermal diffusivity
[A,spgrid] = DiffOp1d(cfun,spgrid,BCtype);

% Neumann boundary input at x=0, sign is based on the
% _outwards_ normal derivative
% This is also affected by the diffusion coefficient at x=0 (according to 
% the approximation in DiffOp1d).
B = sparse([2*cfun(0)/h;zeros(N-1,1)]);

% Disturbance inputs: d_0(t) through 'bd_dist', d_1(t) is an input
% disturbance at x=0, and d_2(t) is a Neumann boundary disturbance at x=1.
Bd0 = bd_dist(spgrid).';
Bd1 = B;
Bd2 = sparse([zeros(N-1,1);2*cfun(1)/h]); % Neumann boundary disturbance at x=1


C = sparse([zeros(1,N-1), 1]); % Measured temperature at x=1

x0 = x0fun(spgrid).';


Sys.A = A;
Sys.B = B;
Sys.Bd = [Bd0,Bd1,Bd2];
Sys.C = C;
Sys.D = 0;
Sys.Dd = [0,0,0];

end

function [Dop,spgrid] = DiffOp1d(cfun,spgrid,BCtype)
% Sparse discrete 1D diffusion operator A defined by (Au)(x)=(c(x)*u'(x))' 
% with Dirichlet or Neumann boundary conditions.
% Inputs:
%   spgrid : The discretized (uniform) spatial grid on the interval [0,L]
%   so that spgrid(1)=0 and spgrid(end)=L, 
%
%   cfun : [1x1 function_handle] Defines the spatially
%   dependent coefficient function c(x)
%
%   BCtype : string Boundary condition configuration, either
%   'DD' for Dirichlet at x=0, Dirichlet at x=1
%   'DN' for Dirichlet at x=0, Neumann at x=1
%   'ND' for Neumann at x=0, Dirichlet at x=1
%   'NN' for Neumann at x=0, Neumann at x=1
% 
% Outputs:
%   Dop : [sparse matrix] The square matrix representing
%   the diffusion operator.
%
%   spgrid : An adjusted spatial grid

N = size(spgrid,2);
assert(N > 0);
h = 1/(N-1);

% We use the approximation:  
% (c(x)u'(x))'(x) ~ 1/h^2*(c(x+h/2)*(u(x+h)-u(x))-c(x-h/2)*(u(x)-u(x-h))

% points x_k+h/2
spmidpoints = 0.5*(spgrid(1:N-1)+spgrid(2:N));
cmid = cfun(spmidpoints);

% Diffusion operator base (Neumann-Neumann case)
cminus = [cmid 0].';
cplus = [0 cmid].';
Dop = spdiags([cminus -(cminus+cplus) cplus],-1:1,N,N);
Dop(1,1) = -(cfun(spgrid(1))+cfun(spgrid(2)));
Dop(1,2) = cfun(spgrid(1))+cfun(spgrid(2));
Dop(N,N) = -(cfun(spgrid(N))+cfun(spgrid(N-1)));
Dop(N,N-1) = cfun(spgrid(N))+cfun(spgrid(N-1));
Dop = 1/h^2*Dop;

% Neumann-Neumann BCs
if isequal(BCtype,'NN')
    % No need to change the grid
    spgrid = spgrid;
elseif isequal(BCtype,'ND')
    spgrid = spgrid(1:N-1);
    Dop = Dop(1:N-1,1:N-1);
elseif isequal(BCtype,'DN')
    spgrid = spgrid(2:N);
    Dop = Dop(2:N,2:N);
elseif isequal(BCtype,'DD')
    spgrid = spgrid(2:N-1);
    Dop = Dop(2:N-1,2:N-1);
else
    error('Unrecognised boundary condition types.')
end

end