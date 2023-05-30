function [w_out] = get_capacitance_approx(epsilon_kappa,epsilon_rho,li,Omega,phase_rho,phase_kappa,delta,C)
    T = 2*pi/Omega;
    N = length(phase_kappa);
    sqrtkappat = @(t) 1./sqrt(1+epsilon_kappa*cos(Omega*t+phase_kappa));
    w3 = @(t) Omega^2/4*(1+((epsilon_kappa^2-1)./(1+epsilon_kappa*cos(Omega*t+phase_kappa)).^2));
    rhot =  @(t) 1./(1 + epsilon_rho*cos(Omega*t+phase_rho));
    M = @(t) makeM(t,delta,li,C,rhot,sqrtkappat,w3);
    [w_out, cnd] = hill_exp(T,M,N);
    [w_out_real,order] = sort(real(w_out),'descend');
    w_out_imag = imag(w_out(order));
    w_out = w_out_real + sqrt(-1).*w_out_imag;
end

function out = makeM(t,delta,li,C,rhot,sqrtkappat,w3)
    Rho = diag(rhot(t));
    Rinv = diag(1./rhot(t));
    K = diag(sqrtkappat(t));
    W3 = diag(w3(t));
    out = delta*diag(1./li)*K*Rho*C*K*Rinv + W3;
end

function [w_out, cnd] = hill_exp(T,M,N)
    W = zeros(2*N,2*N);
    Z = zeros(N,N);
    I = eye(N,N);
    II = eye(2*N,2*N);
    MM = @(t,y) [Z, I; -M(t), Z]*y;
%     for j = 1:N
%         wI0 = zeros(2*N,1); wI0(j) = 1;
%         wII0 = zeros(2*N,1); wII0(N+j) = 1;
%         [~, wI] = ode45(MM, [0,T], wI0);
%         [~, wII] = ode45(MM, [0,T], wII0);
%         W(:,j) = wI(end,:).';
%         W(:,N+j) = wII(end,:).';
%     end
    for j = 1:2*N
        [~, wj] = ode45(MM,[0,T],II(:,j)); 
        W(:,j) = wj(end,:);
    end
    [U, D, V] = eig(W);
    w_out = (log(diag(D))/(1i*T));
    [out_real,ind] = sort(real(w_out),'descend');
    w_out = w_out(ind);
    cnd = cond(U);
end