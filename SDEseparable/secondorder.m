function O=secondorder(B,A,A2,BA,Ibdt,IadW,Ia2dt,IbIadWdt,IaIbdtdW)
    O=B.*Ibdt+A.*IadW+...
      (-A2.*Ia2dt+BA.*(IbIadWdt-IaIbdtdW)).*(.5);
end