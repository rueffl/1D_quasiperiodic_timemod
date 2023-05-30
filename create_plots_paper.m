% main script
format long

%% Static, equidist.

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 1]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2: N
    xm = [xm,li(j)+lij(j)];
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.03; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0; % modulation amplitudes
epsilon_rho = 0;
phase_kappa = [0 pi pi/2]; % modulation phases
phase_rho = [0 pi pi/2];

k_tr = 8; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Fourier coefficients of rhos and kappas
rs = [];
ks = [];
for j = 1:N
    rs_j = [epsilon_rho*exp(-1i*phase_rho(j))/2,1,epsilon_rho*exp(1i*phase_rho(j))/2];
    ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j))/2,1,epsilon_kappa*exp(1i*phase_kappa(j))/2];
    ks = [ks; ks_j];
    rs = [rs; rs_j];
end

% Band functions computation
sample_points = 92;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

 % compute capacitance approximation
for j = 1:sample_points
    alpha = alphas(j);
    C = make_capacitance(N,lij,alpha,L);
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
%     w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
end

% Sort values
w_muller_real = zeros(2*N,sample_points);
w_muller_imag = zeros(2*N,sample_points);
w_cap_real = zeros(2*N,sample_points);
w_cap_imag = zeros(2*N,sample_points);
for j = 1:sample_points
    [w_muller_real(:,j),order] = sort(real(w_muller(:,j)));
    w_muller_imag(:,j) = sqrt(-1).*imag(w_muller(order,j));
    [w_cap_real(:,j),order] = sort(real(w_cap(:,j)));
    w_cap_imag(:,j) = sqrt(-1).*imag(w_cap(order,j));
end

% create plot
figure 
hold on
% plot real part
for i = N+1:2*N
    plot(alphas,w_cap_real(i,:),'.b',markersize=18)
    plot(alphas,w_cap_imag(i,:),'.r',markersize=18)
    if i == 2*N
        legend('Re($\omega_i^{\alpha}$)','Im($\omega_i^{\alpha}$)','Location','northeast',interpreter='latex',fontsize=36)
    end
end
xlabel('$\alpha$',interpreter='latex',fontsize=36)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=36)

%% Static, not equidist.

% main script
format long

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 2]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2: N
    xm = [xm,li(j)+lij(j)];
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.03; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0; % modulation amplitudes
epsilon_rho = 0;
phase_kappa = [0 pi pi/2]; % modulation phases
phase_rho = [0 pi pi/2];

k_tr = 8; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Fourier coefficients of rhos and kappas
rs = [];
ks = [];
for j = 1:N
    rs_j = [epsilon_rho*exp(-1i*phase_rho(j))/2,1,epsilon_rho*exp(1i*phase_rho(j))/2];
    ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j))/2,1,epsilon_kappa*exp(1i*phase_kappa(j))/2];
    ks = [ks; ks_j];
    rs = [rs; rs_j];
end

% Band functions computation
sample_points = 62;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

 % compute capacitance approximation
for j = 1: sample_points
    alpha = alphas(j);
    C = make_capacitance(N,lij,alpha,L);
%     w_cap(:,j) = get_capacitance_approx(epsilon_kappa,epsilon_rho,li,Omega,phase_rho,phase_kappa,delta,C);
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
end

% Sort values
w_cap_real = zeros(2*N,sample_points);
w_cap_imag = zeros(2*N,sample_points);
for j = 1:sample_points
    [w_cap_real(:,j),order] = sort(real(w_cap(:,j)));
    w_cap_imag(:,j) = imag(w_cap(order,j));
end



w_cap_real(end,4:7) = -w_cap_real(1,4:7);
w_cap_real(1,16:19) = -w_cap_real(end,16:19);

% create plot
figure 
hold on
% plot real part
for i = N+1:2*N
    plot(alphas,w_cap_real(i,:),'.b',markersize=18)
    plot(alphas,w_cap_imag(i,:),'.r',markersize=18)
    if i == 2*N
        legend('Re($\omega_i^{\alpha}$)','Im($\omega_i^{\alpha}$)','Location','northeast',interpreter='latex',fontsize=36)
    end
end
xlabel('$\alpha$',interpreter='latex',fontsize=36)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=36)

%% Time-mod, equidist.

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 1]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2: N
    xm = [xm,li(j)+lij(j)];
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.03; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0.2; % modulation amplitudes
epsilon_rho = 0;
phase_kappa = [0 pi pi/2]; % modulation phases
phase_rho = [0 pi pi/2];

k_tr = 8; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Fourier coefficients of rhos and kappas
rs = [];
ks = [];
for j = 1:N
    rs_j = [epsilon_rho*exp(-1i*phase_rho(j))/2,1,epsilon_rho*exp(1i*phase_rho(j))/2];
    ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j))/2,1,epsilon_kappa*exp(1i*phase_kappa(j))/2];
    ks = [ks; ks_j];
    rs = [rs; rs_j];
end

% Band functions computation
sample_points = 92;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

 % compute capacitance approximation
for j = 1:sample_points
    alpha = alphas(j);
    C = make_capacitance(N,lij,alpha,L);
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
%     w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
end

% real(w_cap(end,28:30) = -w_cap(end,28:30);
% real(w_cap(end,63:65)) = -w_cap(end,63:65);

% Sort values
w_cap_real = zeros(2*N,sample_points);
w_cap_imag = zeros(2*N,sample_points);
for j = 1:sample_points
    [w_cap_real(:,j),order] = sort(real(w_cap(:,j)));
    w_cap_imag(:,j) = imag(w_cap(order,j));
end

w_cap_real(1,28:30) = -w_cap_real(1,28:30);
w_cap_real(end,63:65) = -w_cap_real(end,63:65);
for j = 1:sample_points
    [w_cap_real(:,j),order] = sort(w_cap_real(:,j));
    w_cap_imag(:,j) = w_cap_imag(order,j);
end

% create plot
figure 
hold on
for i = N+1:2*N
    plot(alphas,w_cap_real(i,:),'.b',markersize=18)
    plot(alphas,w_cap_imag(i,:),'.r',markersize=18)
    if i == 2*N
        legend('Re($\omega_i^{\alpha}$)','Im($\omega_i^{\alpha}$)','Location','northeast',interpreter='latex',fontsize=36)
    end
end
for i = 1:N
    plot(alphas,w_cap_imag(i,:),'.r',markersize=18)
end
plot(alphas(1).*ones(1,50),linspace(-1*10^(-3),15*10^(-3),50),'--g',linewidth=5)
plot((alphas(7)+alphas(8))/2.*ones(1,50),linspace(-1*10^(-3),15*10^(-3),50),'--g',linewidth=5)
plot((alphas(end-6)+alphas(end-5))/2.*ones(1,50),linspace(-1*10^(-3),15*10^(-3),50),'--g',linewidth=5)
plot((alphas(end-1)+alphas(end))/2.*ones(1,50),linspace(-1*10^(-3),15*10^(-3),50),'--g',linewidth=5)
xlabel('$\alpha$',interpreter='latex',fontsize=36)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=36)



%% Time-mod, not equidist.

% main script
format long

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 2]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2: N
    xm = [xm,li(j)+lij(j)];
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.03; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0.2; % modulation amplitudes
epsilon_rho = 0;
phase_kappa = [0 pi pi/2]; % modulation phases
phase_rho = [0 pi pi/2];

k_tr = 8; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Fourier coefficients of rhos and kappas
rs = [];
ks = [];
for j = 1:N
    rs_j = [epsilon_rho*exp(-1i*phase_rho(j))/2,1,epsilon_rho*exp(1i*phase_rho(j))/2];
    ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j))/2,1,epsilon_kappa*exp(1i*phase_kappa(j))/2];
    ks = [ks; ks_j];
    rs = [rs; rs_j];
end

% Band functions computation
sample_points = 62;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

 % compute capacitance approximation
for j = 1: sample_points
    alpha = alphas(j);
    C = make_capacitance(N,lij,alpha,L);
%     w_cap(:,j) = get_capacitance_approx(epsilon_kappa,epsilon_rho,li,Omega,phase_rho,phase_kappa,delta,C);
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
end

% Sort values
w_cap_real = zeros(2*N,sample_points);
w_cap_imag = zeros(2*N,sample_points);
for j = 1:sample_points
    [w_cap_real(:,j),order] = sort(real(w_cap(:,j)));
    w_cap_imag(:,j) = imag(w_cap(order,j));
end

% create plot
figure 
hold on
% plot real part
for i = N+1:2*N
    plot(alphas,w_cap_real(i,:),'.b',markersize=18)
    plot(alphas,w_cap_imag(i,:),'.r',markersize=18)
    if i == 2*N
        legend('Re($\omega_i^{\alpha}$)','Im($\omega_i^{\alpha}$)','Location','northeast',interpreter='latex',fontsize=36)
    end
end
plot((alphas(8)+alphas(9))/2.*ones(1,50),linspace(-10^(-3),15*10^(-3),50),'--g',linewidth=7)
plot((alphas(12)+alphas(13))/2.*ones(1,50),linspace(-10^(-3),15*10^(-3),50),'--g',linewidth=7)
plot((alphas(58)+alphas(59))/2.*ones(1,50),linspace(-10^(-3),15*10^(-3),50),'--g',linewidth=7)
plot((alphas(43)+alphas(44))/2.*ones(1,50),linspace(-10^(-3),15*10^(-3),50),'--g',linewidth=7)
xlabel('$\alpha$',interpreter='latex',fontsize=36)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=36)


%% Show Muller's Method's Result

k_tr = 6;

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 1]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2:N
    xm = [xm,li(j)+lij(j)];
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.05; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0.4; % modulation amplitudes
epsilon_rho = 0.4;
phase_kappa = [0 pi pi/2]; % modulation phases
phase_rho = [0 pi pi/2];

k_tr = 3; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Fourier coefficients of rhos and kappas
rs = [];
ks = [];
for j = 1:N
    rs_j = [epsilon_rho*exp(-1i*phase_rho(j))/2,1,epsilon_rho*exp(1i*phase_rho(j))/2];
    ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j))/2,1,epsilon_kappa*exp(1i*phase_kappa(j))/2];
    ks = [ks; ks_j];
    rs = [rs; rs_j];
end

% Band functions computation
sample_points = 44;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

for j = 1:sample_points
    alpha = alphas(j);

    % compute capacitance approximation
    if alpha == 0
        C = make_capacitance(N,lij,alpha+0.0000002,L);
        w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
        w_static(:,j)= get_capacitance_approx(0,0,li,Omega,phase_rho,phase_kappa,delta,C);
    else
        C = make_capacitance(N,lij,alpha,L);
        w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
        w_static(:,j)= get_capacitance_approx(0,0,li,Omega,phase_rho,phase_kappa,delta,C);
    end

    if alpha ~= 0
        % compute with mullers method
        for i = 1:2*N
            initial_guess = w_cap(i,j);
    %         w_muller_error(i,j) = minev(getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w_cap(i,j),Omega,rs,ks,vr,delta,v0));
            w_muller(i,j) = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
        end

    end

end

% Broullion zone
for j = 1:sample_points
    for i = 1:2*N
        while w_muller(i,j) > Omega/2
            w_muller(i,j) = w_muller(i,j)-Omega;
        end
        while w_muller(i,j) < -Omega/2
            w_muller(i,j) = w_muller(i,j) + Omega;
        end
    end
end

% Sort values
w_muller_real = zeros(2*N,sample_points);
w_muller_imag = zeros(2*N,sample_points);
w_cap_real = zeros(2*N,sample_points);
w_cap_imag = zeros(2*N,sample_points);
for j = 1:sample_points
    [w_muller_real(:,j),order] = sort(real(w_muller(:,j)));
    w_muller_imag(:,j) = imag(w_muller(order,j));
    [w_cap_real(:,j),order] = sort(real(w_cap(:,j)));
    w_cap_imag(:,j) = imag(w_cap(order,j));
end

% w_muller_real(2,13) = w_cap_real(2,13);
% w_muller_real(2:3,14) = w_cap_real(2:3,14);
% w_muller_real(2:3,19) = w_cap_real(2:3,19);
% w_muller_real(2,20) = w_cap_real(2,20);
w_muller_real(5,10) = w_cap_real(5,10);
w_muller_real(:,13) = w_cap_real(:,13);

figure()
hold on
for i = N+1:2*N
    plot(alphas,w_muller_real(i,:),'-b',linewidth=3,markersize=4)
    plot(alphas,w_cap_real(i,:),'--c',linewidth=3,markersize=4)
    plot(alphas,w_muller_imag(i,:),'-r',linewidth=3,markersize=4)
    plot(alphas,w_cap_imag(i,:),'--g',linewidth=3,markersize=4)
    if i == 2*N
        legend('Muller`s Method (real part)','Capacitance Approximation (real part)','Muller`s Method (imaginary part)','Capacitance Approximation (imaginary part)', ...
            fontsize=20, location='southoutside')
    end
end
xlabel('$\alpha$',interpreter='latex',fontsize=18)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=18)



figure()
hold on
for i = 1:2*N
    plot(alphas,real(w_muller(i,:)),'ob',linewidth=1.2,markersize=4)
    plot(alphas,real(w_cap(i,:)),'*c',linewidth=1.2,markersize=4)
    plot(alphas,imag(w_muller(i,:)),'or',linewidth=1.2,markersize=4)
    plot(alphas,imag(w_cap(i,:)),'*m',linewidth=1.2,markersize=4)
    if i == 2*N
        legend('Muller`s Method (real part)','Capacitance Approximation (real part)','Muller`s Method (imaginary part)','Capacitance Approximation (imaginary part)', ...
            fontsize=14, location='southoutside')
    end
end
xlabel('$\alpha$',interpreter='latex',fontsize=14)
ylabel('$\omega_i^{\alpha}$',interpreter='latex',fontsize=14)










