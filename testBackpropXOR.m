X = [ 0 0 1; 0 1 1; 1 0 1; 1 1 1;];
D = [ 0 1 1 0 ];

W1 = 2* rand(4, 3) -1;
W2 = 2* rand(1, 4) -1;

for epoch = 1:1000
    [W1 W2] = BackpropXOR(W1, W2, X, D);
end

N = 4;

for k = 1:N
    x = X(k, :)';
    v1 = W1 * x;
    y1 = Sigmoid(v1);
    v2 = W2 * y1;
    y2(k) = Sigmoid(v2);
end

disp('Results = '), disp(y2)
