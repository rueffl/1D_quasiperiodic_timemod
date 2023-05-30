%% Set Parameter Values
format long

% Settings for the structure
N = 2; % number of the resonator
li = [1 1]; % length of the resonators
lij = [1 1]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

if N > 1
    for j = 2:N
        xm = [xm,li(j)+lij(j)];
    end
end

xp = xm + li; 
delta = 0.0001; % high contrast parameter

% Settings for modulation
Omega = 0.05; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0.4; % modulation amplitudes
epsilon_rho = 0.4;
phase_kappa = [0 pi]; % modulation phases
phase_rho = [0 pi];

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
sample_points = 22;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);

%% Compare Run Time

Ks = [1:2:9];
times = zeros(1,length(Ks));

c = 1;
for k_tr = Ks

    for j = 1: sample_points
        alpha = alphas(j);
        
        % compute static case
        C = make_capacitance(N,lij,alpha,L);
        w_static(:,j)= get_capacitance_approx(0,0,li,Omega,phase_rho,phase_kappa,delta,C);
    
        % compute with mullers method
        for i = 1:2*N
            initial_guess = w_static(i,j);
            tic
            w_muller(i,j) = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
            times(c) = times(c) + toc;
        end
    end
    c = c + 1;

end

figure()
semilogx(Ks,times,'*--')
ylabel('Runtime [s]',interpreter='latex',fontsize=36)
xlabel('$K$',interpreter='latex',fontsize=36)

%% Investigate Relative Error

Ks = [1:5,8,12:14];
rel_error = zeros(1,length(Ks));
abs_error = zeros(1,length(Ks));

c = 1;
for k_tr = Ks

    for j = 1: sample_points
        alpha = alphas(j);
        
        % compute static case
        C = make_capacitance(N,lij,alpha,L);
    
        % compute capacitance approximation
        w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
    
        % compute with mullers method
        for i = 1:2*N
            initial_guess = w_cap(i,j);
            w_muller(i,j) = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
            w_muller_error(i,j) = minev(getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w_muller(i,j),Omega,rs,ks,vr,delta,v0));
        end
    
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

    rel_error(c) = max(max((sqrt(((w_muller_real-w_cap_real)).^2+((w_muller_imag-w_cap_imag)).^2)/sqrt(w_muller_real.^2+w_muller_imag.^2))'));
    abs_error(c) = max(max((sqrt(((w_muller_real-w_cap_real)).^2+((w_muller_imag-w_cap_imag)).^2))'));
    c = c+1;

end


figure()
loglog(Ks(1:5),abs_error(1:5),'x--')
ylabel('Absolute Error',interpreter='latex',fontsize=14)
xlabel('$K$',interpreter='latex',fontsize=14)


%% Iterate over N for fixed K

delta = 0.0001; % high contrast parameter
Omega = 0.05; % modulation frequency
T = 2*pi/Omega;
epsilon_kappa = 0.4; 
epsilon_rho = 0.4;
phase_kappa = [0]; 
phase_rho = [0];
v0 = 1;
vr = 1;

k_tr = 3;
Ns = [1:6];

times_muller = zeros(1,length(Ns));
times_cap = zeros(1,length(Ns));
c = 1;
for N = Ns
    
    li = ones(1,N); % length of the resonators
    lij = ones(1,N); % distance between the resonators
    L = sum(li)+sum(lij); % length of the unit cell
    % define the boundary points x_minus and x_plus
    xm = [0];
    
    if N > 1
        for j = 2:N
            xm = [xm,li(j)+lij(j)];
        end
    end
    xp = xm + li; 
    
    phase_kappa = [phase_kappa, pi/N];
    phase_rho = [phase_rho, pi/N];
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
    sample_points = 22;
    alphas = linspace(-pi/L,pi/L,sample_points);
    
    % initializations
    w_static = zeros(2*N,sample_points);
    w_cap = zeros(2*N,sample_points);
    w_muller = zeros(2*N,sample_points);
    
    for j = 1: sample_points
        alpha = alphas(j);
        
        % compute static case
        tic;
        C = make_capacitance(N,lij,alpha,L);
        % compute capacitance approximation
        w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
        times_cap(c) = times_cap(c) + toc;
        % compute with mullers method
        for i = 1:2*N
            initial_guess = w_cap(i,j);
            tic;
            w_muller(i,j) = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
            times_muller(c) = times_muller(c) + toc;
        end
    
    end

    c = c+1;

end

figure()
plot(Ns,times_muller,'*--b')
hold on
plot(Ns,times_cap,'*--r')
ylabel('Runtime [s]',interpreter='latex',fontsize=36)
xlabel('$N$',interpreter='latex',fontsize=36)



