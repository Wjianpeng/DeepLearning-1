function y = Conv(x, W)
%UNTITLED4 이 함수의 요약 설명 위치
%   자세한 설명 위치
    [wrow, wcol, numFilters] = size(W);
    [xrow, xcol, ~] = size(x);
    
    yrow = xrow - wrow + 1;
    ycol = xcol - wcol + 1;
    
    y = zeros(yrow, ycol, numFilters);
    
    for k = 1:numFilters
        filter = W(:, :, k);
        filter = rot90(squeeze(filter), 2);
        y(:, :, k) = conv2(x, filter, 'valid');
    end
end

