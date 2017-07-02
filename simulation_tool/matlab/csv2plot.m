function [systime i1 di0 di1] = csv2plot( plot_name )

fprintf('nazwa pliku: %s\n',plot_name);
fontsize=16;

data = csvread([plot_name,'.csv']);
systime = data(:,1)./ 1000.;
i1 = data(:,4);
di0 = data(:,15);
di1 = data(:,16);


% figure(1), grid on,  hold on
% [hAx,hLine1,hLine2] = plotyy(systime, i1,systime, di0);
% set(hLine1, 'LineWidth', 1, 'Color','r');
% set(hLine2, 'LineWidth', 1, 'Color','k');
% line(systime,di1,'parent',hAx(2),'LineWidth', 1, 'Color','b')
% xlabel('time [s]', 'FontSize',fontsize)
% set(get(hAx(1), 'YLabel'), 'String', 'current [mA]','FontSize',fontsize)
% set(get(hAx(2), 'YLabel'), 'String', 'marker state', 'FontSize',fontsize)
% % set(gca,'FontSize',fontsize)
% ylim(hAx(2),[0 1.01])
% ylim(hAx(1),[0 700])
% % xlim(hAx(1),[0 700])
% % axes(hAx,[0 700 0 1.01])
% print(plot_name,'-dpng')


end

