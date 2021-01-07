function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
bias=ones(m,1);
X=[bias X];

Z2=X*Theta1';
a2=sigmoid(Z2);
bias2=ones(size(Z2,1),1);
a2=[bias2 a2];
Z3=a2*Theta2';
a3=sigmoid(Z3);

% You need to return the following variables correctly 
%p = zeros(size(X, 1), 1);
[max_value, p] = max(a3, [], 2);








% =========================================================================


end
