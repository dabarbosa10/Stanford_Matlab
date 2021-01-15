function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%================     FeedForward    ========================================
bias1=ones(m,1);
A1=[bias1, X];
Z2=A1*Theta1';
A2=sigmoid(Z2);
%========================
bias2=ones(m,1);
A2=[bias2, A2];
Z3=A2*Theta2';
A3=sigmoid(Z3);

%========================= Encoding Vectors ==============================
Encode=eye(num_labels);
%========================= Supplementary variables ========================
a=log(A3);
b=log(1-A3);
%========================= Computing cost =================================
cost=0;
for i=1:m
   cost=cost-(a(i,:)*Encode(y(i),:)'+b(i,:)*(1-Encode(y(i),:)'))  ;
end
%==========================================================================
%==========================================================================
%===================    Regularization   ==================================
The1=Theta1(:,2:end);
The2=Theta2(:,2:end);
a1=sum((The1(:)).^2);
a2=sum((The2(:)).^2);
reg=(a1+a2)/(2*m);
J=(1/m)*cost+lambda*reg;
%==========================================================================
delta_3=zeros(m,num_labels);
gprime=[ones(m,1) sigmoidGradient(Z2)];
delta_2=zeros(m,hidden_layer_size+1);
%%=================    Backprop ============================================
for t=1:m
    delta_3(t,:)=A3(t,:)-Encode(y(t),:);
    delta_2(t,:)=(delta_3(t,:)*Theta2).*gprime(t,:);    
end

Theta1_grad=(1/m)*(Theta1_grad+(delta_2(:,2:end))'*A1);
Theta2_grad=(1/m)*(Theta2_grad+(delta_3)'*A2);
%================ Regularization ==========================================
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
