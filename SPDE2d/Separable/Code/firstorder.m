function O=firstorder(B1,B2,A,t,Ibdt,IadW)
    O=B1.*t+B2.*Ibdt+A.*IadW;
end