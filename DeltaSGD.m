function W = DeltaSGD(W, X, D)
%UNTITLED 이 함수의 요약 설명 위치
%   Delta Stochastic Gradient Descent 
    alpha = 0.5;
    
    N = 4;
    
    for k = 1:N
        x = X(k, :)';
        d = D(k);
        
        v = W*x;
        y = Sigmoid(v);
        
        e = d - y;
        delta = y * (1-y) * e;
        
        dW = alpha * delta * x;
        
        W(1) = W(1) + dW(1);
        W(2) = W(2) + dW(2);
        W(3) = W(3) + dW(3);
    end 
%    disp('dW1 = '),disp(dW(1)), disp(', dW2 = '), disp(dW(2)), disp(', dW3='), disp(dW(3))

end

