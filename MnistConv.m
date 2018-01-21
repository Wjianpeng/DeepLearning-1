function [ W1, W5, W0 ] = MnistConv(W1, W5, W0, X, D)
%UNTITLED7 이 함수의 요약 설명 위치
%   자세한 설명 위치
    alpha = 0.01;
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dW0 = zeros(size(W0));
    
    N = 28 * 28;
    for k = 1:N
        x = X(:,:,k);
        d = D(k);
        
        y1 = Conv(x, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5 * y4;
        y5 = ReLU(v5);
        v6 = W0 * y5;
        y6 = Softmax(v6);
        
        e6 = d - y6;
        delta6 = e6;
        e5 = W0' * delta6;
        delta5 = (y5 > 0) .* e5;
        e4 = W5' * delta5;
        delta4 = (y4 > 0) .* e4;
        e3 = reshape(e4, size(y3));
        e2 = zeros(size(y2));
        W3 = ones(size(y2))/(2*2);
        for c = 1:20
            e2(:,:,c) = kron(e3(:,:,c),ones([2 2])) .* W3(:,:,c);
        end
        delta2 = (y2 > 0) .* e2;
        delta1_x = zeros(size(W1));
        for c = 1:20
            delta1_x(:,:,c) = conv2(x(:,:), rot90(delta2(:,:,c),2),'valid');
        end
        
        dW1 = dW1 + alpha * delta1_x;
        dW5 = dW5 + alpha * delta5 * y4';
        dW0 = dW0 + alpha * delta6 * y5';
    end
    W1 = W1 + dW1;
    W5 = W5 + dW5;
    W0 = W0 + dW0;
end

