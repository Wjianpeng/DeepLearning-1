clear all

X = [0 0 1;
     0 1 1;
     1 0 1;
     1 1 1;];
 
D = [0 0 1 1];
 
W = 2 * rand(1,3) - 1; % -1 ~ 1 

disp('before W');
W
for epoch = 1:10000
    W = DeltaSGD(W, X, D);
end
disp('after W');
W

N = 4;
for k = 1: N
    x = X(k, :)';
    v = W * x;
    yy(k) = Sigmoid(v);
end

disp('Last y:')
yy
