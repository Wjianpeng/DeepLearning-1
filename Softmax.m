function y = Softmax( x )
%UNTITLED7 이 함수의 요약 설명 위치
%   자세한 설명 위치
    ex = exp(x);
    y = ex / sum(ex);
end

