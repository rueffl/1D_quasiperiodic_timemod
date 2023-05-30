%% Set Parameter Values
format long

% Settings for the structure
N = 3; % number of the resonator
li = [1 1 1]; % length of the resonators
lij = [1 1 2]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
% define the boundary points x_minus and x_plus
xm = [0];

for j = 2:N
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

k_tr = 5; % truncation parameters as in remark 3.3
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
sample_points = 82;
alphas = linspace(-pi/L,pi/L,sample_points);

%% Plot Band Functions

% initializations
w_static = zeros(2*N,sample_points);
w_cap = zeros(2*N,sample_points);
w_cap_old = zeros(2*N,sample_points);
w_muller = zeros(2*N,sample_points);
w_muller_error = zeros(2*N,sample_points);

for j = 1: sample_points
    alpha = alphas(j);
    
    % compute static case
    C = make_capacitance(N,lij,alpha,L);
    w_static(:,j)= get_capacitance_approx(0,0,li,Omega,phase_rho,phase_kappa,delta,C);

    % compute capacitance approximation
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
    w_cap_old(:,j) = get_capacitance_approx_spec_rho(epsilon_kappa,phase_kappa,epsilon_rho,phase_rho,Omega,delta,li,C);

    % compute with mullers method
    % check degeneracy 
    for i = 1:N*2
        initial_guess = w_cap(i,j);
        w_muller_error(i,j) = minev(getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w_cap(i,j),Omega,rs,ks,vr,delta,v0));
        w_muller(i,j) = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
    end

end

% Broullion zone
for j = 1: sample_points
    for i = 1:N*2
        while w_muller(i,j) > Omega/2
            w_muller(i,j) = w_muller(i,j)-Omega;
        end
        while w_muller(i,j) < -Omega/2
            w_muller(i,j) = w_muller(i,j) + Omega;
        end
    end
end

% plot real parts
figure 
hold on
for i = 1:2*N
    plot(alphas,real(w_static(i,:)),'r')
    plot(alphas,real(w_cap(i,:)),'.r')
    plot(alphas,real(w_cap_old(i,:)),'*r')
    plot(alphas,real(w_muller(i,:)),'ob')
    legend('$\omega_i^{\alpha}$ static','$\omega_i^{\alpha}$ cap','$\omega_i^{\alpha}$ cap old','$\omega_i^{\alpha}$ muller','Location','southeast',interpreter='latex')
    title(strcat('Band functions (real) for $\varepsilon_{\kappa}$=',num2str(epsilon_kappa),'$,\,\varepsilon_{\rho}$=',num2str(epsilon_rho),'$,\,\Omega$=',num2str(Omega),'$,\,K$=',num2str(k_tr)),Interpreter='latex')    
    xlabel('$\alpha$',Interpreter='latex')
    ylabel('$\omega_i^{\alpha}$',interpreter='latex')
end

% plot imaginary parts
figure 
hold on
for i = 1:2*N
    plot(alphas,imag(w_static(i,:)),'g')
    plot(alphas,imag(w_cap(i,:)),'.r')
    plot(alphas,imag(w_cap_old(i,:)),'*r')
    plot(alphas,imag(w_muller(i,:)),'ob')
    legend('$\omega_i^{\alpha}$ static','$\omega_i^{\alpha}$ cap','$\omega_i^{\alpha}$ cap old','$\omega_i^{\alpha}$ muller','Location','southeast',interpreter='latex')
    title(strcat('Band functions (imaginary) for $\varepsilon_{\kappa}$=',num2str(epsilon_kappa),'$,\,\varepsilon_{\rho}$=',num2str(epsilon_rho),'$,\,\Omega$=',num2str(Omega),'$,\,K$=',num2str(k_tr)),Interpreter='latex') 
    xlabel('$\alpha$',Interpreter='latex')
    ylabel('$\omega_i^{\alpha}$',interpreter='latex')
end


%% Create Surface Plot

p = @(w,alpha) svds(getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w,Omega,rs,ks,vr,delta,v0),1,"smallest");
% ws = linspace(-Omega+10^(-4),Omega-10^(-4),200);
ws = linspace(-Omega/2,Omega/2,200);
alphas = linspace(-pi/L,pi/L,100);
ps = zeros(100,50);
for i = 1:200
    for j = 1:100
        ps(i,j) = p(ws(i),alphas(j));
    end
end
figure()
surface(alphas,ws,ps)
colorbar
xlabel('$\alpha$',Interpreter='latex',fontsize=14)
ylabel('$\omega$',Interpreter='latex',fontsize=14)
title(strcat('Smallest eigenvalue of $\mathcal{A}$ for $\varepsilon_{\kappa}= $ ',num2str(epsilon_kappa),', $\varepsilon_{\rho}= $ ',num2str(epsilon_rho),', $\Omega= $ ',num2str(Omega),', $K =$ ',num2str(k_tr)),Interpreter="latex")    

sample_points = 50;
alphas = linspace(-pi/L,pi/L,sample_points);
for j = 1:sample_points
    alpha = alphas(j)
    
    % compute static case
    C = make_capacitance(N,lij,alpha,L);
    w_static(:,j) = get_capacitance_approx_spec(0,zeros(N,1),Omega,delta,li,C);
    w_cap(:,j) = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C);
    w_cap_old(:,j) = get_capacitance_approx(epsilon_kappa,epsilon_rho,li,Omega,phase_rho,phase_kappa,delta,C);

end
hold on
for i = 1:sample_points
    plot3(alphas(i).*ones(2*N,1),w_cap(:,i),ones(2*N,1),'r*')
end
for i = 1:sample_points
    plot3(alphas(i).*ones(2*N,1),w_static(:,i),ones(2*N,1),'go')
end
for i = 1:sample_points
    plot3(alphas(i).*ones(2*N,1),w_cap_old(:,i),ones(2*N,1),'ro')
end




