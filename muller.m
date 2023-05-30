function xplus = muller(initial_guess,alpha,N,lij,L,xm,xp,k_tr,Omega,rs,ks,vr,delta,v0)
%   define three initial guesses
    xnm2 = initial_guess*(1+0.01);
    xnm1 = initial_guess*(1+0.01*exp(1i*2*pi/3));
    xn = initial_guess;
%   tolerance for exiting the mullers method
    tol = 1e-12;

%   iteration number
    i = 0;
    xplus = xn;

%   function to iterate with
    p = @(w) minev(getMatcalA(alpha,N,lij,L,xm,xp,k_tr,w,Omega,rs,ks,vr,delta,v0));
    
%   implementation of mullers method
    while (abs(p(xn)) > tol)

        q = (xn - xnm1)/(xnm1 - xnm2);
        a = q*p(xn) - q*(1+q)*p(xnm1) + q^2*p(xnm2);
        b = (2*q + 1)*p(xn) - (1+q)^2*p(xnm1) + q^2*p(xnm2);
        c = (1 + q)*p(xn);
                
        r = xn - (xn - xnm1)*((2*c)/(b + sqrt(b^2 - 4*a*c)));
        s = xn - (xn - xnm1)*((2*c)/(b - sqrt(b^2 - 4*a*c)));
        
        if(abs(p(r)) < abs(p(s)))
            xplus = r;
            else
            xplus = s;
        end

        xnm2 = xnm1;
        xnm1 = xn;
        xn = xplus;
        
        i = i + 1;
        disp(strcat('alpha=',num2str(alpha),': Iteration  ',num2str(i),' xn=',num2str(xn),' error=',num2str(p(xn))))
    end
    
end