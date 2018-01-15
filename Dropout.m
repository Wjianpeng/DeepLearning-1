function ym = Dropout(y, ratio)
%UNTITLED10 이 함수의 요약 설명 위치
%   자세한 설명 위치
    [m, n] = size(y);
    ym = zeros(m, n);
    
    num = round(m*n*(1-ratio));
    idx = randperm(m*n, num);
    ym(idx) = 1 / (1 - ratio);
end

