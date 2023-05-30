%% Numerically compute the Floquet exponents of the NxN system of Capacitance ODEs
%  GCM\Psi + d/dt 1/\kappa d/dt \Psi = 0. 1/kappa has a finite Fourier series of length 2
%
% Rewrite to the system of 1st order ODEs
% dt/dt \psi_1 = \kappa*psi_2
% dt/dt \psi_2 = -GCM*psi_1 
% Then solve spectrally: d/dt = 1i(\omega + n\Omega) which gives the
% eigenvalue problem
% \omega \psi_1 = - n\Omega \psi_1 - 1i*\kappa*\psi_2
% \omega \psi_2 = - n\Omega \psi_2 - 1i*GCM*\psi_1
% 

function w_out = get_capacitance_approx_spec(epsilon_kappa,phase_kappa,Omega,delta,li,C)
    GCM = delta*diag(1./li)*C;

    M = 1; % Number of Fourier coefficients of 1/\kappa
    N = size(GCM,1);
    N_fourier = 4; % Length of Fourier series approx
    
    K_mod = zeros(2*M+1,N);
    for i = 1:N
        K_mod(:,i) = [epsilon_kappa/2*exp(-1i*phase_kappa(i)); 1; epsilon_kappa/2*exp(1i*phase_kappa(i))]; % Fourier coefficients of 1/\kappa
    end

    ns = -N_fourier:N_fourier;

    NN = 2*N_fourier+1;
    O = diag(ns.'*Omega);
    e = ones(NN,1);
    INN = eye(NN);
    IN = eye(N);
    iK = zeros(NN*N);
    for i = 1:N
        K = zeros(NN,NN);
        for m = -M:M
            K = K+diag(e(1:NN-abs(m))*K_mod(m+M+1,i),m);
        end
        Ii = (i-1)*NN+1:i*NN;
        iK(Ii,Ii) = inv(K); %% Fourier coefficients of \kappa
    end

    Z = zeros(NN*N);
    mat = -[kron(IN,O), Z; Z, kron(IN,O)]  - 1i*[Z, iK; -kron(GCM,INN), Z]; % Kroenecker product to get the RHS matrix

    w_out = eigs(mat,2*N,'smallestabs'); % The eigenvalues of "mat" are approximately \omega + n\Omega for |n| < N_fouier. Taking the smallest eigenvalues corresponds to n = 0.
    w_out = sort(w_out);
end