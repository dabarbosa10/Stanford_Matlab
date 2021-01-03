function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;


figure; hold on;
class1=find(y==1);
class2=find(y==0);

plot(X(class1,1),X(class1,2),'+k','MarkerSize',7, 'LineWidth',2);
plot(X(class2,1),X(class2,2),'o','MarkerSize',7, 'MarkerEdgeColor','k','MarkerFaceColor','y');
grid();  
xlabel('Exam 1 Score');
ylabel('Exam 2 Score');
legend('Admitted','Not Admitted')
hold off; 


end
