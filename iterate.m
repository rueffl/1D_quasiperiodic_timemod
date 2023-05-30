format long

% Settings for the structure
N = 2; % number of the resonator
li = [1,1]; % length of the resonators
lij = [1,2]; % distance between the resonators
L = sum(li)+sum(lij); % length of the unit cell
xm = [0,li(1)+lij(1)]; % define the boundary points x_minus and x_plus
xp = xm + li; 
delta = 0.000001; % high contrast parameter
k_tr = 2; % truncation parameters as in remark 3.3
vr = 1;
v0 = 1;

% Settings for modulation
Omega = 0.01;
T = 2*pi/Omega;
epsilon_rho = 0;
all_epsilon_kappa = [0.4];
phase_kappa = [0,pi/2]; 
phase_rho = [0,pi/2];

% Band functions computation
sample_points = 10;
alphas = linspace(-pi/L,pi/L,sample_points);

% initializations

figure()
hold on
for c = 1:length(all_epsilon_kappa)
    epsilon_kappa = all_epsilon_kappa(c);
    rs = [];
    ks = [];
    for j = 1:N
        rs_j = [epsilon_rho*exp(-1i*phase_rho(j)),1,epsilon_rho*exp(-1i*phase_rho(j))];
        ks_j = [epsilon_kappa*exp(-1i*phase_kappa(j)),1,epsilon_kappa*exp(-1i*phase_kappa(j))];
        ks = [ks; ks_j];
        rs = [rs; rs_j];
    end

    w_static = zeros(2*N,sample_points);
    w_cap = zeros(2*N,sample_points);
    w_muller = zeros(N,sample_points);
    its = zeros(N,sample_points);

    for j = 1:sample_points
        alpha = alphas(j);
        
        % compute static case
        C = make_capacitance(N,lij,alpha,L);
        w_static(:,j)= get_capacitance_approx(0,0,li,Omega,phase_rho,phase_kappa,delta,C);
    
        % compute capacitance approximation
        w_cap(:,j) = get_capacitance_approx(epsilon_kappa,epsilon_rho,li,Omega,phase_rho,phase_kappa,delta,C);
    
        for i = 1:N
            initial_guess = w_static(i,j);
%             if (i == 2 && j == 2) | (i == 2 && j == sample_points-1)
%                 initial_guess = 1.2*10^(-3);
%             end
            [w_muller(i,j), it] = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0);
%             if w_muller(i,j) < 0
%                 w_muller(i,j) = -1*w_muller(i,j);
%             end
        end
    
    end
    
    subplot(2,3,c)
    hold on
    for i = 1:N
        plot(alphas,real(w_static(i,:)),'g')
        plot(alphas,real(w_cap(i,:)),'.r')
        plot(alphas,real(w_muller(i,:)),'ob')
        legend('$\omega$ static','$\omega$ cap','$\omega$ muller','Location','southeast',interpreter="latex")
        title(strcat('Band functions for $\varepsilon_{\kappa}= $ ',num2str(epsilon_kappa),', $\varepsilon_{\rho}= $ ',num2str(epsilon_rho),', $\Omega= $ ',num2str(Omega),', $K =$ ',num2str(k_tr)),Interpreter="latex")    
        xlabel('$\alpha$',Interpreter="latex")
        ylabel('Quasifrequencies')
    end
end