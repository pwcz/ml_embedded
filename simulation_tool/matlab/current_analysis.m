
clc, close all

[t, itx , y1, y2 ]= csv2plot('<file name>');

itx = smooth(itx, 'moving');


figure(1), hold on
% index = find(y1 ==0 & y2 == 0 & itx < 5);
% index = find(y1 ==0 & y2 == 0 & itx > 7);
index = find(y1 ==1 & y2 == 0 );  % rejestracja
% index = find(y1 ==1 & y2 == 1  ); % transmisja

[hAx,hLine1,hLine2] = plotyy(t, itx,t,y1);
set(hLine1, 'LineWidth', 2, 'Color','r','LineStyle','-');
set(hLine2, 'LineWidth', 1, 'Color','k','LineStyle','-');
line(t,y2,'parent',hAx(2),'LineWidth', 1, 'Color','k')

filly = zeros([size(itx),1]);
filly(index) = 1000;

harea = area(t,filly, 'LineStyle', 'none');
set(harea, 'FaceColor', 'b')
alpha(0.25)
hold on

avg_current = mean(itx(index));
fprintf('sredni prad = %f\n',avg_current);
time_on_action = sum(ii_time(2:2:length(ii_time)) - ii_time(1:2:length(ii_time)));


liczba_transmisji = size(find(logical(diff(y2)) > 0))./2;
fprintf('liczba_transmisji = %d\n',liczba_transmisji(1));
time_on_action = time_on_action ./ liczba_transmisji(1);
fprintf('sredni czas akcji = %f\n',time_on_action);
