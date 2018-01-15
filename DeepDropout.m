function [ W1, W2, W3, W4 ] = DeepDropout(W1, W2, W3, W4, X, D)
%UNTITLED4 이 함수의 요약 설명 위치
%   자세한 설명 위치
    alpha = 0.01;
    
    N = 5;
    for k = 1:N
        x = reshape(X(:, :, k), 25, 1);
        d = D(k, :)';
        
        v1 = W1 * x;
        y1 = Sigmoid(v1);
        y1 = y1 .* Dropout(y1, 0.2);
        v2 = W2 * y1;
        y2 = Sigmoid(v2);
        y2 = y2 .* Dropout(y2, 0.2);
        v3 = W3 * y2;
        y3 = Sigmoid(v3);
        y3 = y3 .* Dropout(y3, 0.2);
        v4 = W4 * y3;
        y4 = Softmax(v4);
        
        e4 = d - y4;
        delta4 = e4;
        e3 = W4' * delta4;
        delta3 = y3 .* (1-y3) .* e3;
        e2 = W3' * delta3;
        delta2 = y2 .* (1-y2) .* e2;
        e1 = W2' * delta2;
        delta1 = y1 .* (1-y1) .* e1;
        
        dW1 = alpha * delta1 * x';
        W1 = W1 + dW1;
        dW2 = alpha * delta2 * y1';
        W2 = W2 + dW2;
        dW3 = alpha * delta3 * y2';
        W3 = W3 + dW3;
        dW4 = alpha * delta4 * y3';
        W4 = W4 + dW4;
    end


end


