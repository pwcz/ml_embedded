% Script which presents current consumption during SMS transmission

clc, close all

[t, itx , y1, y2 ]= csv2plot('00_SMS_matlab_short_presentation');


itx = smooth(itx, 'moving');


figure(1), hold on

index = find(y1 ==0 & y2 == 0 & itx < 5);
index2_p = find(y1 ==0 & y2 == 0 & itx > 7);
index2 = index2_p(1:33);
index3_p = find(y1 ==1 & y2 == 0 );  % rejestracja
index4 = [find(y1 ==1 & y2 == 1  ); index3_p(112:end) ; index2_p(34:end)]; % transmisja
index3 = index3_p(1:111);
plot(t,itx,'-b','LineWidth',3)

ylabel('pobor pradu [mA]','Fontsize',18)
xlabel('czas [s]','Fontsize',18)

filly = zeros([size(itx),1]);
filly(index) = 1000;
harea = area(t,filly, 'LineStyle', 'none');
set(harea, 'FaceColor', 'b')
alpha(0.50)

filly = zeros([size(itx),1]);
filly(index2) = 1000;
harea = area(t,filly, 'LineStyle', 'none');
set(harea, 'FaceColor', 'g')
alpha(0.50)

filly = zeros([size(itx),1]);
filly(index3) = 1000;
harea = area(t,filly, 'LineStyle', 'none');
set(harea, 'FaceColor', 'y')
alpha(0.50)

filly = zeros([size(itx),1]);
filly(index4) = 1000;
harea = area(t,filly, 'LineStyle', 'none');
set(harea, 'FaceColor', 'r')
alpha(0.50)

ii = find(logical(diff(filly)) > 0);
filly = ones([size(ii),1]).*1000;
plot(t(ii),filly,'ko');

legend('pobor pradu','stan uspienia','wlaczenie zasilania peryferiow', 'wlaczenie modemu i rejestracja do sieci','transmisja' ,'Location','best')
axis([0,30,0,500])

ii_time = t(ii);
time_on_action = sum(ii_time(2:2:length(ii_time)) - ii_time(1:2:length(ii_time)));
