function [Gnj,Vnj] = getG_and_V(xm,xp,n,k_tr,rs,fis,list_lambdas)
    N = length(xm);
    Gnj=zeros(2*N);
    Vnj = zeros(2*N);    
    
    for i = 1 : N
        fi = fis(:,i);
        lambda = list_lambdas(i);
    %   Build blocks of Gnj    
        sum = 0;
        for m = -1:1
            if abs(n-m)>k_tr
                sum = sum;
            else
                a = rs(i,m+2);
                b = fi(k_tr+1-n+m);
                sum = sum + a*b;
            end
        end
        Gi = sum* [-1i*lambda*exp(1i*lambda*xm(i)) 1i*lambda*exp(-1i*lambda*xm(i));
            1i*lambda*exp(1i*lambda*xp(i)) -1i*lambda*exp(-1i*lambda*xp(i))];
        Gnj(2*i-1:2*i,2*i-1:2*i)=Gi;
    %   Build blocks of Vnj
    Vi = fi(k_tr+1-n)* [exp(1i*lambda*xm(i)) exp(-1i*lambda*xm(i));
                         exp(1i*lambda*xp(i)) exp(-1i*lambda*xp(i))];
    Vnj(2*i-1:2*i,2*i-1:2*i)=Vi;        
    end
end