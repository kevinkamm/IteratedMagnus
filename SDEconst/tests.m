d=100;
M=100;
A=5.*ones(d,d);
B=randn(d,d,1,M);
timeit(@()A.*B)
timeit(@()2.*B)

% tic;
% toc;
timeit(@()switchTest(1,0))
function switchTest(x,y)
if size(x,4)==1
switch sprintf('%d%d',x,y)
    case '00'
    case '01'
    case '11'
    case '10'
end
end
end