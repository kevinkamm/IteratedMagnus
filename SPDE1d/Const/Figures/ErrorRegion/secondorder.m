function O=secondorder(B,A,A2,BA,t,W,IW)
    O=B.*t+A.*W-.5.*A2.*t+BA.*IW-.5.*BA.*t.*W;
end