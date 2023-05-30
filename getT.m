function T_matrix = getT(kn,alpha,N,lij,L)
    T_matrix = zeros(2*N);
    T_matrix(1,1) = -(kn*cos(kn*lij(end)))/sin(kn*lij(end));
    T_matrix(1,end) = kn/sin(kn*lij(end))*exp(-1i*alpha*L);
    T_matrix(end,1) = kn/sin(kn*lij(end))*exp(1i*alpha*L);
    T_matrix(end,end) = -kn*cos(kn*lij(end))/sin(kn*lij(end));
    for j = 1:N-1
        T_matrix(2*j:2*j+1,2*j:2*j+1) = makeAl(kn,lij(j));
    end
end

function Al = makeAl(kn,l)
    Al = kn*[-cos(kn*l)/sin(kn*l), 1/sin(kn*l);
          1/sin(kn*l),-cos(kn*l)/sin(kn*l)];
end