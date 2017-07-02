clc, close all
[t, itx , v1 ]= csv2plot('<file_name>');

start = find(v1 > 6, 1);

N = 45;

result = zeros(N,1);

for R = 1:N
    stop =  find(t > (t(start) + 60*60), 1) ;
    mean_voltage = mean(v1(start:stop));
    mean_current = mean(itx(start:stop))./1000;
    power = mean_current * mean_voltage * (t(stop) - t(start));
    result(R) = power;
    start = stop;
end

result