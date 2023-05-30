function A = define_A(alpha,N,lij,L,xm,xp,k_tr,w,Omega,rs,ks,vr,delta,v0)
%DEFINE_A Constructs the matrix \mathcal{A}^* as defined by (37)
%   Detailed explanation goes here

    A = zeros(2*N*(2*k_tr+1),2*N*(2*k_tr+1));

    for i = 1:N

        Ci =  getC(k_tr,i,w,Omega,rs,ks,vr);
        [fi,lambdas_tilde] = eig(Ci,'vector');
        lambdas = sqrt(lambdas_tilde);
        
        for n = k_tr:-1:-k_tr

            An = [];
            kn = (w+n*Omega)/v0;
            T_matrix = getT(kn,alpha,N,lij,L);

            j = n+k_tr+1;
            sum = 0;
            for m = -1:1
                if abs(n-m)>k_tr
                    sum = sum;
                else
                    a = rs(i,m+2);
                    b = fi(k_tr+1-n+m,j);
                    sum = sum + a*b;
                end
            end

            Gi = sum* [-1i*lambdas(j)*exp(1i*lambdas(j)*xm(i)) 1i*lambdas(j)*exp(-1i*lambdas(j)*xm(i));
                         1i*lambdas(j)*exp(1i*lambdas(j)*xp(i)) -1i*lambdas(j)*exp(-1i*lambdas(j)*xp(i))];
            Vi = fi(k_tr+1-n,j)* [exp(1i*lambdas(j)*xm(i)) exp(-1i*lambdas(j)*xm(i));
                         exp(1i*lambdas(j)*xp(i)) exp(-1i*lambdas(j)*xp(i))];

            

        end
        
        [h,b] = size(An);
%         A(,:) = An;

    end

end

function MatcalA = getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w,Omega,rs,ks,vr,delta,v0)
    MatcalA = [];
    for n = k_tr:-1:-k_tr
        An = [];

%  define Dirchlet Neumann map
        kn = (w+n*Omega)/v0;
        T_matrix = getT(kn,alpha,N,lij,L);

        for k = 1:k_tr*2+1
%           assemble matrix An (eq35)
            [Gnk,Vnk] = getG_and_V(xm,xp,n,k,k_tr,w,Omega,rs,ks,vr);
            An = [An Gnk-delta*T_matrix*Vnk];
        end
%   assemble matrix MatcalA (eq37)
    MatcalA = [MatcalA; An];
    end
end

function [Gnj,Vnj] = getG_and_V(xm,xp,n,j,k_tr,w,Omega,rs,ks,vr)
    N = length(xm);
    Gnj = zeros(2*N);
    Vnj = zeros(2*N);    
    
    for i = 1 : N
        Ci =  getC(k_tr,i,w,Omega,rs,ks,vr);
        [fi,lambdas_tilde] = eig(Ci,'vector');
        lambdas = sqrt(lambdas_tilde);
    %   Build blocks of Gnj    
        sum = 0;
        for m = -1:1
            if abs(n-m)>k_tr
                sum = sum;
            else
                a = rs(i,m+2);
                b = fi(k_tr+1-n+m,j);
                sum = sum + a*b;
            end
        end
        Gi = sum* [-1i*lambdas(j)*exp(1i*lambdas(j)*xm(i)) 1i*lambdas(j)*exp(-1i*lambdas(j)*xm(i));
            1i*lambdas(j)*exp(1i*lambdas(j)*xp(i)) -1i*lambdas(j)*exp(-1i*lambdas(j)*xp(i))];
        Gnj(2*i-1:2*i,2*i-1:2*i)=Gi;
    %   Build blocks of Vnj
    Vi = fi(k_tr+1-n,j)* [exp(1i*lambdas(j)*xm(i)) exp(-1i*lambdas(j)*xm(i));
                         exp(1i*lambdas(j)*xp(i)) exp(-1i*lambdas(j)*xp(i))];
    Vnj(2*i-1:2*i,2*i-1:2*i)=Vi;        
    end
end