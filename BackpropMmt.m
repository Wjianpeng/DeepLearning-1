function [ W1, W2 ] = BackpropXOR(W1, W2, X, D)
%UNTITLED2 이 함수의 요약 설명 위치
%   XOR 문제를 Back Propagation으로 해결
    alpha = 0.9;
    beta = 0.9;
    
    mmt1 = zeros(size(W1));
    mmt2 = zeros(size(W2));
    
    N = 4;
    
    for k = 1:N
        x = X(k, :)';
        d = D(k);
        
        v1 = W1 * x;
        y1 = Sigmoid(v1);
        
        v2 = W2 * y1;
        y2 = Sigmoid(v2);
        
        v = v2; y = y2;
        
        e2 = d - y2;
        delta2 = y2 .* (1 - y2) .* e2;
        
        e1 = W2' * delta2;
        delta1 = y1 .* (1 - y1) .* e1;
        
        dW1 = alpha * delta1 * x';
        mmt1 = dW1 + beta * mmt1;
        W1 = W1 + mmt1;
        
        dW2 = alpha * delta2 * y1';
        mmt2 = dW2 + beta * mmt2;
        W2 = W2 + mmt2;
        
    end
    

end

