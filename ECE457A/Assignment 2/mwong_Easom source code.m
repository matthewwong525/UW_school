
OF = @objective_function;
a = -99;
b = 99;
lb = [-100 -100];
ub = [100 100];
x_init = rand(10,2,1)*(b-a) + a;
for n = 1:10
    [x,f,exitFlag,output] = simulannealbnd(OF,x_init(n,:),lb,ub)
end
a = 0.25;
b = 1;
t_init = rand(10,1)*(b-a) + a;
init_x = [4.23 3.45]
for n = 1:10
    options = optimoptions('simulannealbnd', 'InitialTemperature', t_init(n));
    [x,fval,exitFlag,output] = simulannealbnd(OF,init_x,lb,ub, options)
end



