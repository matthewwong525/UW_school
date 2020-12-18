function y = objective_function(x)
x1 = x(1);
x2 = x(2);
y = -1.*cos(x1).* cos(x2).* exp(-1.*(x1-pi).^2-(x2 -pi).^2);
end