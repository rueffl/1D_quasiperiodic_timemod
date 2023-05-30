% get the smalles eigenvalue of the matrix A
function ev_min = minev(A)
    ev = eig(A);
    [ev_abs,order] = sort(abs(ev),'ascend');
    ev_min = ev(order(1));
end