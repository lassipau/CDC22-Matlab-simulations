%% Robust control of a 2D heat equation on a rectangle with boundary input, 
% output and boundary disturbance. 
% 
% The system is a heat equation with constant heat conductivity 1 on a
% square domain Omega=[0,1]x[0,1]. The system has Neumann boundary
% conditions.
% The input u(t), the output y(t) and disturbance w_d(t) are scalar-valued.
% 
% The Neumann boundary input u(t) acts on the part of the boundary where 
% 0<x<1/2 and y=0.
%
% The output is the average temperature over the part of the boundary where
% 0<x<1 and y=1.
%
% The Neumann boundary disturbance w_d(t) acts on the part of the boundary
% where x=0 and 0<y<1
% 
% 
% The system is unstable, but its only unstable eigenvalue 0 can be 
% stabilized using LQR/LQG design.
%
% The controller construction uses the RORPack Matlab library (available 
% freely at https://github.com/lassipau/rorpack-matlab/ ). The library 
% needs to be installed to run the code (their directories and 
% subdirectories should be included in the Matlab PATH).

N = 31; 

% Initial state of the plant
x0fun = @(x,y) zeros(size(x));
%x0fun = @(x,y) 0.5*(1+cos(pi*(1-x))).*(1-1/4*cos(2*pi*y));
% x0fun = @(x,y) 0.5*(1+cos(pi*(1-x)));
%x0fun = @(x,y) 1/2*x.^2.*(3-2*x)-1;
%x0fun = @(x,y) 1/2*x.^2.*(3-2*x)-1/2;
%x0fun = @(x,y) 1*(1-x).^2.*(3-2*(1-x))-1;
%x0fun = @(x,y) .5*(1-x).^2.*(3-2*(1-x))-.5;
%x0fun = @(x,y) 1/4*(x.^3-1.5*x.^2)-1/4;
%x0fun = @(x,y) .2*x.^2.*(3-2*x)-.5;

[x0,spgrid,Sys] = ConstrHeat2D_CDC22(N,N,x0fun,1);

dimX = size(Sys.A,1);
dimU = size(Sys.B,2);
dimY = size(Sys.C,1);

% Define the reference and disturbance signals, and list the
% required frequencies in 'freqsReal'

% Case 1:
yref = @(t) cos(pi*t);
wdist = @(t) .2*ones(size(t))+3*cos(3*pi*t+1);
freqsReal = [0, pi,3*pi];

% Case 2:
% yref = @(t) ones(size(t));
% wdist = @(t) ones(size(t));
% freqsReal = 0;

% Case 3:
% yref = @(t) sin(2*t)+.1*cos(6*t);
% wdist = @(t) sin(t);
% freqsReal = [0,1,2,6];


% Check the consistency of the system definition
Sys = SysConsistent(Sys,yref,wdist,freqsReal);

%% Construct the controller

% Construct another Finite Difference approximation for controller design

N_lo = 24;
[~,spgrid_lo,Sys_lo] = ConstrHeat2D_CDC22(N_lo,N_lo,x0fun,1);


% K21 and L are chosen based on LQR/LQG design 

dimX_lo = size(Sys_lo.A,1);

IMstabmarg = 0.5;
K0 = -lqr_RP(Sys_lo.A+IMstabmarg*eye(dimX_lo),Sys_lo.B,10*eye(dimX_lo),0.001*eye(dimU));
% Alternative: Using Control System Toolbox
% K0 = -lqr(Sys_lo.A+IMstabmarg*eye(dimX_lo),Sys_lo.B,100*eye(dimX_lo),0.001*eye(dimU),zeros(dimX_lo,dimU));

L_lo = -lqr_RP(Sys_lo.A'+IMstabmarg*eye(dimX_lo),Sys_lo.C',10*eye(dimX_lo),0.001*eye(dimY))';
% L_lo = -lqr(Sys_lo.A'+IMstabmarg*eye(dimX_lo),Sys_lo.C',100*eye(dimX_lo),0.001*eye(dimY),zeros(dimX_lo,dimU))';


max(real(eig(Sys_lo.A+Sys_lo.B*K0)))
max(real(eig(Sys_lo.A+L_lo*Sys_lo.C)))

% % Construct the internal model. The system has scalar output, dimY=1

[G1,G2] = ConstrIM(freqsReal,dimY);
dimZ = size(G1,1);

%% Compute B1, K1 and H_K

% Compute the values P_K(iw_k)
PKappr = @(s) (Sys_lo.C+Sys_lo.D*K0)*((s*eye(size(Sys_lo.A,1))-Sys_lo.A-Sys_lo.B*K0)\Sys_lo.B)+Sys_lo.D;
PKvals = cell(1,length(freqsReal));
for ind = 1:length(freqsReal)
   PKvals{ind} = PKappr(1i*freqsReal(ind));
end


B1 = [];
for ind = 1:length(freqsReal)
    
    if freqsReal(ind) == 0
        B1 = [B1;PKvals{ind}];
    else
        B1 = [B1;real(PKvals{ind});-imag(PKvals{ind})];
    end
end

% Construct K1 using LQR

K1 = -lqr_RP(G1+IMstabmarg*eye(dimZ),B1,10*eye(dimZ),0.001*eye(dimU));
% K1 = -lqr(G1+IMstabmarg*eye(dimZ),B1,100*eye(dimZ),0.001*eye(dimU),zeros(dimZ,dimU));


% Compute H_K as an (approximate) solution of the Sylvester equation 
% G1*H_K = H_K*(A+B*K21)+G2*(C_Lambda+D*K21)
H_K = sylvester(G1,-(Sys_lo.A+Sys_lo.B*K0),G2*(Sys_lo.C+Sys_lo.D*K0));


% Choose K2 with the formula K2=K21+K1*H_K
K2_lo = K0+K1*H_K;

%%
% Interpolate K2 and L to the original spatial grid
K2_square = reshape(K2_lo,N_lo,N_lo);
K2 = interp2(spgrid_lo.xx,spgrid_lo.yy,K2_square,spgrid.xx,spgrid.yy);
K2 = K2(:).';

L_square = reshape(L_lo,N_lo,N_lo);
L = interp2(spgrid_lo.xx,spgrid_lo.yy,L_square,spgrid.xx,spgrid.yy);
L = L(:);

% % Check that [K1,K2] stabilizes the pair (As,Bs)  and L stabilizes (C,A)
% As = [G1,G2*Sys.C;zeros(dimX,dimZ),Sys.A];
% Bs = [G2*Sys.D;Sys.B];
% 
% max(real(eig(As+Bs*[K1,K2])))
% max(real(eig(Sys.A+L*Sys.C)))


% Construct the Observer-Based Robust Controller 
ContrSys = ObserverBasedRC(freqsReal,Sys,[K1,K2],L,'full_K');

% % % Alternative controller choices from RORPack
% 
% K0_square = reshape(K0,N_lo,N_lo);
% K0_hi = interp2(spgrid_lo.xx,spgrid_lo.yy,K0_square,spgrid.xx,spgrid.yy);
% K0_hi = K0_hi(:).';
% ContrSys = ObserverBasedRC(freqsReal,Sys,K0_hi,L,'LQR',1);
% % ContrSys = ObserverBasedRC(freqsReal,Sys,K0_hi,L,'poleplacement',1.5);
% % ContrSys = DualObserverBasedRC(freqsReal,Sys,K0_hi,L,'poleplacement',1.5);
 

%% Closed-loop construction and simulation

% Construct the closed-loop system
CLSys = ConstrCLSys(Sys,ContrSys);

% Print an approximate stability margin of the closed-loop system
stabmarg = CLStabMargin(CLSys)

% Define the initial state of the closed-loop system
% (the controller has zero initial state by default).
xe0 = [x0;zeros(size(ContrSys.G1,1),1)];

% Set the simulation length and define the plotting grid
Tend = 16;
tgrid = linspace(0,Tend,300);

% Simulate the closed-loop system
CLsim = SimCLSys(CLSys,xe0,yref,wdist,tgrid,[]);

%% Visualization

% Plot the (approximate) eigenvalues of the closed-loop system
figure(1)
PlotEigs(CLSys.Ae,[-20 .3 NaN NaN]);

% Choose whther or not to print titles of the figures
PrintFigureTitles = true;

% Plot the controlled outputs, the tracking error norm, and 
% the control inputs
figure(2)
subplot(3,1,1)
PlotOutput(tgrid,yref,CLsim,PrintFigureTitles)
subplot(3,1,2)
PlotErrorNorm(tgrid,CLsim,PrintFigureTitles)
subplot(3,1,3)
PlotControl(tgrid,CLsim,PrintFigureTitles)

%% State of the controlled PDE

figure(3)
% plotind = 155;
plotind = 147;

PlotHeat2DSurfCase2(CLsim,spgrid,tgrid,plotind)
xlabel('$\xi_1$','Interpreter','latex','Fontsize',20)
ylabel('$\xi_2$','Interpreter','latex','Fontsize',20)
%% Animation of the state of the controlled PDE

figure(4)
% colormap jet
% No movie recording
[~,zlims] = AnimHeat2DCase2(CLsim,spgrid,tgrid,0.03,0);

% Movie recording
% [MovAnim,zlims] = AnimHeat2DCase2(CLsim,spgrid,tgrid,0,1);

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
