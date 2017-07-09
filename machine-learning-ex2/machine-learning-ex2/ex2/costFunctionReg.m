function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
theta_tmp=theta; 
theta_tmp(1)=0; %計算用、最初のθのみ0にする。
[J_ini, grad_ini] = costFunction(theta, X, y);

J = J_ini +(lambda/(2*m)*sum(theta_tmp.^2));



grad = grad_ini+lambda/m*theta_tmp;
%grad(1) = grad_ini(1);
% =============================================================

end
