function [x0,spgrid,Sys] = ConstrHeat2D_CDC22(N,M,x0fun,cval)
% Construct the system operators for the 2D heat equation example. 
% Approximation with a Finite Difference scheme.
% The control input, the disturbance input, and
% the measured output act on distinct parts of the boundary of
% the rectangle.
%
% The Neumann boundary input u(t) acts on the part of the boundary where 
% 0<x<1/2 and y=0.
%
% The output is the average temperature over the part of the boundary where
% 0<x<1 and y=1.
%
% The Neumann boundary disturbance w_d(t) acts on the part of the boundary
% where x=0 and 0<y<1

xx = linspace(0,1,N);
yy = linspace(0,1,M);
[xxg,yyg] = meshgrid(xx,yy);
spgrid.xx = xxg;
spgrid.yy = yyg;

x0 = reshape(x0fun(xxg,yyg),N*M,1);

if nargout > 2
  dx = 1/(N-1);
  dy = 1/(M-1);

  ex = ones(N,1);
  ey = ones(M,1);

  Dxx = spdiags([ex -2*ex ex],[-1:1],N,N);
  Dyy = spdiags([ey -2*ey ey],[-1:1],M,M);
  
  % Neumann boundary conditions: v_0=v_1, v_{N+1}=v_N
  Dxx(1,1) = -1; 
  Dxx(end,end) = -1;
  Dyy(1,1) = -1;
  Dyy(end,end) = -1;
  
  A = cval * full(kron(Dyy, speye(N)) + kron(speye(M), Dxx))/(dx*dy);
  % Inputs and outputs

  % Neumann boundary control input, the part of the boundary where 0<x<1/2 and y=0
  % (corresponds to "first column" of x in grid form
  B = reshape(2/dx*[[ones(floor(N/2),1); zeros(N-floor(N/2),1)] zeros(N,M-1)],N*M,1);

  
  % Observation is the integral over the part of the boundary where y=1
  % corresponding indices are N*(N-1)+1..N
  C = zeros(1,N*M);
  C(1,(N-1)*M+1:end) = dx;
  C(1,(N-1)*M+1) = dx / 2; 
  C(1, end) = dx / 2;
  
  
  D = 0;
  
  % Disturbance signal input operator, Neumann input from the boundary where x=0 and 0<y<1
  Bd = zeros(N*M,1);
  Bd(1:N:N*M,1) = 1/dy;
  
  
  Sys.A = A;
  Sys.B = B;
  Sys.Bd = Bd;
  Sys.C = C;
  Sys.D = D;
  Sys.Dd = zeros(size(Sys.C,1),size(Sys.Bd,2));
end

