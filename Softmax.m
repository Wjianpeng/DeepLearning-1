function y = Softmax( x )
%UNTITLED7 �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ
    ex = exp(x);
    y = ex / sum(ex);
end

