function [ W1, W2 ] = MultiClassNumber( W1, W2, X, D )
%UNTITLED 이 함수의 요약 설명 위치
%   자세한 설명 위치
    alpha = 0.01;
    
    N = 5;
    for k = 1:N
        x = reshape(X(:,:,k),25,1)
        d = D(k)';
        v1 = W1 * x;
        y1 = Sigmoid(v1);
        
        v2 = W2 * y1;
        y2 = Softmax(v2);
        
        e2 = d - y2;
        delta2 = e2;
        
        e1 = W2' * delta2;
        delta1 = y1 .* (1-y1) .* e1;
        
        dW1 = alpha * delta1 * x';
        W1 = W1 + dW1;
        
        dW2 = alpha * delta2 * y1';
        W2 = W2 + dW2;
    end

end

