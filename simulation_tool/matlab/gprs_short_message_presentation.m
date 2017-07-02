% Script which presents current consumption during GPRS transmission
clc, close all

[t, itx , y1, y2 ]= csv2plot('02_REST_1U_A_matlab_short_presentation');

t = t - t(1);

itx = smooth(itx, 'moving');

figure(1), hold on

index = find(y1 ==0 & y2 == 0 & itx < 5);
index2_p = find(y1 ==0 & y2 == 0 & itx > 7);
index2 = index2_p(1:32);
index3_p = find(y1 ==1 & y2 == 0  );  % rejestracja
index3 = index3_p(1:125);
index4 = [find(y1 ==1 & y2 == 1  );index2_p(33:end);index3_p(126:end)];

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
axis([0,30,0,600])

