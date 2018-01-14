function [ W1, W2 ] = BackpropXOR(W1, W2, X, D)
%UNTITLED2 �� �Լ��� ��� ���� ��ġ
%   XOR ������ Back Propagation���� �ذ�
    alpha = 0.9;
    
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
        W1 = W1 + dW1;
        
        dW2 = alpha * delta2 * y1';
        W2 = W2 + dW2;
        
    end
    

end

