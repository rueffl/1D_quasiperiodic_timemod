function C = make_capacitance(N,lij,alpha,L)
      C = zeros(N);
    for i = 1:N
            for j = 1: N
                if i == j - 1
                    C(i, j) = C(i, j) - 1 / lij(i);
                end
                if i == j
                    if j-1 <1
                        C(i, j) = C(i, j) + (1 / lij(N) + 1 / lij(j));
                        else
                        C(i, j) = C(i, j) + (1 / lij(j - 1) + 1 / lij(j));
                    end
                end
                if i == j + 1
                    C(i, j) =  C(i, j) - 1 / lij(j);
                end
                if (j == 1) && (i == N)
                    C(i, j) = C(i, j)- exp(1j * alpha * L) / lij(N);
                end
                if (i == 1) && (j == N)
                    C(i, j) =  C(i, j) - exp(-1j * alpha * L) / lij(N);
                end
            end
    end
end