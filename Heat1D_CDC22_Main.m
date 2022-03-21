%% Heat equation on the interval [0,1] with 
% Neumann boundary control and Dirichlet boundary observation 
% Approximation with a Finite differences scheme 
% 
% Numerical example for the CDC 2022 manuscript by L. Paunonen and J.-P.
% Humaloja.
%
% Neumann boundary control at x=0, regulated output y(t) and a 
% Neumann boundary disturbance at x=1
% Unstable system, stabilization by stabilizing the only unstable
% eigenvalue =0
%
% The controller construction uses the RORPack Matlab library (available 
% freely at https://github.com/lassipau/rorpack-matlab/ ) and the Chebfun
% Matlab library (available freely at https://www.chebfun.org/ ). Both
% libraries need to be installed to run the code (their directories and 
% subdirectories should be included in the Matlab PATH).


% Size of the Finite Difference approximation
N = 101;

% Initial state of the heat equation
%x0fun = @(x) zeros(size(x));
x0fun = @(x) 0.5*(1+cos(pi*(1-x)));
%x0fun = @(x) 0.5*(1+cos(pi*(1-x)));
%x0fun = @(x) 1/2*x.^2.*(3-2*x)-1;
%x0fun = @(x) 1/2*x.^2.*(3-2*x)-1/2;
%x0fun = @(x) 1*(1-x).^2.*(3-2*(1-x))-1;
%x0fun = @(x) 0.5*(1-x).^2.*(3-2*(1-x))-0.5;
%x0fun = @(x) 1/4*(x.^3-1.5*x.^2)-1/4;
%x0fun = @(x) 0.2*x.^2.*(3-2*x)-0.5;

% The spatially varying thermal diffusivity of the material
% cfun = @(t) ones(size(t));
% cfun = @(t) 1+t;
% cfun = @(t) 1-2*t.*(1-2*t);
cfun = @(t) 1+0.5*cos(5/2*pi*t);
% cfun = @(t) 1+sin(pi*t);
% cfun = @(t) 0.3-0.6*t.*(1-t);

% Distributed disturbance input profile.
bd_dist = @(x) x.*(1-x);

% Construct the system.
[x0,Sys,spgrid,BCtype] = ConstrHeat1D_CDC22(cfun,bd_dist,x0fun,N);


% Define the reference and disturbance signals, and list the
% required frequencies in 'freqsReal'

%yref = @(t) sin(2*t)+.1*cos(6*t);
%yref = @(t) sin(2*t)+.2*cos(3*t);
%yref = @(t) ones(size(t));

%wdist = @(t) ones(size(t));
%wdist = @(t) sin(t);
%wdist = @(t) zeros(size(t));

yref = @(t) 0.5+cos(2*t)+0.2*cos(3*t+0.4);%+.2*cos(3*t);
% wdist = @(t) zeros(size(t));
wdist = @(t) [1+2*cos(2*t+0.5);cos(5*t+1);2*cos(3*t)+cos(5*t)];

% yref = @(t) ones(size(t));
% wdist = @(t) ones(size(t));

% yref = @(t) sin(2*t)+.1*cos(6*t);
% wdist = @(t) sin(t);

% List the (nonnegative) frequencies of yref and wdist in 'freqsReal'
freqsReal = [0, 1, 2, 3, 5];


% Check the consistency of the system definition
Sys = SysConsistent(Sys,yref,wdist,freqsReal);

%% Construct the controller

% An observer-based robust controller 
% Stabilizing state feedback and output injection operators K_{21} and L
% These are chosen based to correspond to bounded operators. We choose
% K21x = -<x,1>_X = \int_0^1 x(\xi)d\xi
% L = -1(.)

% Only the single unstable
% eigenvalue at s=0 needs to be stabilized
K21 = -1/(N-1)*ones(1,N);
% PlotEigs(full(Sys.A+Sys.B*K),[-20 1 -.3 .3])

L = -ones(N,1);
% PlotEigs(full(Sys.A+L*Sys.C),[-20 1 -.3 .3]);

% Construct the internal model (G1,G2). The system has scalar output, dimY=1
[G1,G2] = ConstrIM(freqsReal,1);

%%
% Compute the required transfer function values P_K(iw_k) and P_{KI}(iw_k) using Chebfun: This
% can be done by solving a boundary value problem related to each
% frequency, and evaluating the solution at x=0.

% First compute the values for P_K(iw_k) (the values P_K(-iw_k) are
% conjugates)
PKvals = cell(1,length(freqsReal));
for ind = 1:length(freqsReal)
    s = freqsReal(ind);
    cb_A = chebop(0,1);
    cb_A.op = @(x,w) 1i*s*w-diff(cfun(x)*diff(w));
%     cb_A.lbc = @(w) diff(w)+1;

    cb_A.lbc = @(w) diff(w)-sum(w)+1;
    cb_A.rbc = @(w) diff(w);
    w = cb_A\0;
    
    PKvals{ind} = w(1);
end

% % Alternative: Compute P_K(iw) using the original Finite Difference
% % approximation of (A,B,C,D).
% Pappr = @(s) (Sys.C+Sys.D*K21)*((s*eye(size(Sys.A,1))-Sys.A-Sys.B*K21)\Sys.B)+Sys.D;
% Pvals = cell(1,length(freqsReal));
% for ind = 1:length(freqsReal)
%    Pvals{ind} = Pappr(1i*freqsReal(ind));
% end


B1 = [];
for ind = 1:length(freqsReal)
    
    if freqsReal(ind) == 0
        B1 = [B1;PKvals{ind}];
    else
        B1 = [B1;real(PKvals{ind});-imag(PKvals{ind})];
    end
end
        

% Compute the values P_{KI}(iw_k)\phi_n for the orthonormal basis \psi_0=1, \psi_n(\xi)=sqrt(2)*cos(pi*n*\xi)

% The truncation index for the approximation of H_K
N_H = 30;

% Choose the orthonormal basis {\psi_n}_n of L^2(0,1)
psi_n = @(xi,n) (rem(n,2)==0)*((n>0)*sqrt(2)+(n==0)*1)*cos(2*pi*(n/2)*xi) + (rem(n,2)==1)*sqrt(2)*sin(2*pi*(floor(n/2)+1)*xi);
% psi_n = @(xi,n) ((n>0)*sqrt(2)+(n==0)*1)*cos(pi*n*xi);
% psi_n = @(xi,n) sqrt(2)*sin(pi*(n+1)*xi);

PKIvals = cell(1,length(freqsReal));
for ind = 1:length(freqsReal)
    s = freqsReal(ind);
    
    cb_A = chebop(0,1);
    cb_A.op = @(x,w) 1i*s*w-diff(cfun(x)*diff(w));

    cb_A.lbc = @(w) diff(w)-sum(w);
    cb_A.rbc = @(w) diff(w);
    
    tmp = zeros(1,N_H+1);
    for ind_n = 0:N_H
        cb_psi_n = chebfun(@(xi) psi_n(xi,ind_n),[0,1]);
        w = cb_A\cb_psi_n;
        
        tmp(ind_n+1) = w(1);
    end

    PKIvals{ind} = tmp;
end


% Compute the approximation of the matrix H_K
Psi_proj = zeros(N_H+1,length(spgrid));

for ind_n = 0:N_H
    Psi_proj(ind_n+1,:) = 1/(N-1)*psi_n(spgrid,ind_n);
end

Psi_mat = (N-1)*Psi_proj';


H_K_basis = [];
for ind = 1:length(freqsReal)
    
%     HK_element = PKIvals{ind}*Psi_proj
    if freqsReal(ind) == 0
        H_K_basis = [H_K_basis;PKIvals{ind}];
    else
        H_K_basis = [H_K_basis;real(PKIvals{ind});-imag(PKIvals{ind})];
    end
end
H_K = H_K_basis*Psi_proj;



% % Alternative: Compute H_K as the solution of a Sylvester equation using
% % the original Finite Difference approximation.
% H_K = sylvester(G1,-(Sys.A+Sys.B*K21),G2*(Sys.C+Sys.D*K21));


%% Construct the controller


% Choose K1 so that G1+B1*K1 is Hurwitz
dimZ = size(G1,1);
dimU = 1;
IMstabmarg = 0.45;
K1 = -lqr_RP(G1+IMstabmarg*eye(dimZ),B1,10*eye(dimZ),0.001*eye(dimU));
% Alternative: Using Control System Toolbox
% K1 = -lqr(G1+IMstabmarg*eye(dimZ),B1,100*eye(dimZ),0.001*eye(dimU),zeros(dimZ,dimU));
 
% Choose K2 with the formula K2=K21+K1*H_K
K2 = K21+K1*H_K;


ContrSys = ObserverBasedRC(freqsReal,Sys,[K1,K2],L,'full_K');

%% Closed-loop construction and simulation

% Construct the closed-loop system
CLSys = ConstrCLSys(Sys,ContrSys);

% Print an approximate stability margin of the closed-loop system
stabmarg = CLStabMargin(CLSys)

% Define the initial state of the closed-loop system
% (the controller has zero initial state by default).
xe0 = [x0;zeros(size(ContrSys.G1,1),1)];

% Set the simulation length and define the plotting grid
Tend = 14;
tgrid = linspace(0,Tend,300);

% Simulate the closed-loop system
CLsim = SimCLSys(CLSys,xe0,yref,wdist,tgrid,[]);

%% Visualization

% Choose whether or not to print titles of the figures
PrintFigureTitles = true;

% Plot the (approximate) eigenvalues of the closed-loop system
figure(1)
PlotEigs(CLSys.Ae,[-20 .3 -6.2 6.2])

% Plot the controlled outputs, the tracking error norm, and 
% the control inputs
figure(2)
subplot(3,1,1)
PlotOutput(tgrid,yref,CLsim,PrintFigureTitles)
box off
grid on
subplot(3,1,2)
PlotErrorNorm(tgrid,CLsim,PrintFigureTitles)
subplot(3,1,3)
PlotControl(tgrid,CLsim,PrintFigureTitles)

%% State of the controlled PDE

figure(3)

plotskip = 3;
Plot1DHeatSurf(CLsim.xesol(1:plotskip:N,:),spgrid(1:plotskip:end),tgrid,BCtype)

%% Animation of the state of the controlled PDE

figure(4)
% No movie recording
[~,zlims] = Anim1DHeat(CLsim.xesol(1:N,:),spgrid,tgrid,BCtype,0.03,0);

% Movie recording
% [MovAnim,zlims] = Anim1DHeat(CLsim.xesol(1:N,:),spgrid,tgrid,BCtype,0.03,1);

%movie(MovAnim)

%% Plot the reference signal

figure(5)
tt = linspace(0,16,500);
plot(tt,yref(tt),'Color',1.1*[0 0.447 0.741],'Linewidth',3);
title('Reference signal $y_{ref}$','Interpreter','latex','Fontsize',16)
set(gca,'xgrid','on','ygrid','on','tickdir','out','box','off')


%% Export movie to AVI

%AnimExport = VideoWriter('heat2Danim.avi','Uncompressed AVI');
% AnimExport = VideoWriter('Case1-animation.mp4','MPEG-4');
% AnimExport = VideoWriter('Case2-animation.mp4','MPEG-4');
% AnimExport = VideoWriter('Case3-animation.mp4','MPEG-4');
% AnimExport.Quality = 100;

% AnimExport = VideoWriter('Case1-animation.avi','Uncompressed AVI');
% AnimExport = VideoWriter('Case2-animation.avi','Uncompressed AVI');
% AnimExport = VideoWriter('Case3-animation.avi','Uncompressed AVI');

% AnimExport.FrameRate = 15;
% open(AnimExport);
% writeVideo(AnimExport,MovAnim);
% close(AnimExport);
