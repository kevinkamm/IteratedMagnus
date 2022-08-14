function [exact,eulerRef,euler,m1,m2,m3,m4,varargout]=methodColors()
exactColor=[0 0 0]; % black
eulerRefColor=[0.6350 0.0780 0.1840]; % default red
eulerColor=[0.8500 0.3250 0.0980]; % default orange
m1Color=[0 0.4470 0.7410]; % default dark blue
m2Color=[0.9290 0.6940 0.1250]; % default yellow
m3Color=[0.4660 0.6740 0.1880]; % default green
m4Color=[0.4940 0.1840 0.5560]; % default purple

exact.linestyle={'-','-'};
exact.marker={'none','none'};
exact.color={exactColor,rgb2gray(exactColor)};

euler.linestyle={'none','none'};
euler.marker={'*','*'};
euler.color={eulerRefColor,rgb2gray(eulerRefColor)};

eulerRef.linestyle={'none','none'};
eulerRef.marker={'*','*'};
eulerRef.color={eulerColor,rgb2gray(eulerColor)};

m1.linestyle={'none','none'};
m1.marker={'.','.'};
m1.color={m1Color,rgb2gray(m1Color)};

m2.linestyle={'none','none'};
m2.marker={'o','o'};
m2.color={m2Color,rgb2gray(m2Color)};

m3.linestyle={'none','none'};
m3.marker={'x','x'};
m3.color={m3Color,rgb2gray(m3Color)};

m4.linestyle={'none','none'};
m4.marker={'x','x'};
m4.color={m4Color,rgb2gray(m4Color)};

if nargout>7
    inputNameDict.Exact='exact';
    inputNameDict.EulerRef='euler ref';
    inputNameDict.Euler='euler';
    inputNameDict.Magnus1='m1';
    inputNameDict.Magnus2='m2';
    inputNameDict.Magnus3='m3';
    inputNameDict.Magnus4='m4';
    varargout{1}=inputNameDict;
end
if nargout>8
    colorDict.Exact=exact;
    colorDict.EulerRef=eulerRef;
    colorDict.Euler=euler;
    colorDict.Magnus1=m1;
    colorDict.Magnus2=m2;
    colorDict.Magnus3=m3;
    colorDict.Magnus4=m4;
    varargout{2}=colorDict;
end
end
%% Matlab default colors
% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]
%% Matlab in-built colors
% 'red'     'r'	[1 0 0]
% 'green'	'g'	[0 1 0]
% 'blue'	'b'	[0 0 1]
% 'cyan'	'c'	[0 1 1]
% 'magenta'	'm'	[1 0 1]
% 'yellow'	'y'	[1 1 0]
% 'black'	'k'	[0 0 0]
% 'white'	'w'	[1 1 1]