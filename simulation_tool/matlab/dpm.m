% Script which plot ilustration of DPM 
clc, clear all, close all

x1 = [0 5 10 15];
y1 = ones(size(x1));

x2 = [17 19 21 23 ];
y2 = ones(size(x2));
user_x = [7 20.5];
user_y = [1 1];

figure(1), hold on
for X = 1:size(x1')
    rectangle('Position',[x1(X),0,1,1],'FaceColor','r')
end
for X = 1:size(x2')
    rectangle('Position',[x2(X),0,1,1],'FaceColor','b')
end

first_action = .44;
first_action_2 = 0.347;
font_size = 32;
annotation('textarrow',[0.2,first_action_2 - 0.005 ], [0.6,0.525],'String','','FontSize',font_size)
annotation('doublearrow',[first_action_2,first_action], [0.55,0.55], 'Color','b')
annotation('line', [first_action_2 first_action_2], [0.525 .55],'LineStyle','--')
annotation('line', [first_action first_action], [0.515 .55],'LineStyle','--')

stem(user_x, user_y,'g','filled')
text(7, 1.18,'','Color','b','FontSize',font_size)
axis off

axis([0 25 0 2]);