function [systime i1 v1 ] = csv2plot( plot_name )
  fprintf('nazwa pliku: %s\n',plot_name);

  data = csvread([plot_name,'.csv']);
  systime = data(:,1)./ 1000.;
  i1 = data(:,4);
  v1 = data(:,8);
end
