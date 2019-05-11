function loss = nneval(nn, data_x, data_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 3, 'Wrong number of arguments');

nn.testing = 1;
% dataset performance
nn     = nnff(nn, data_x, data_y);
loss.e = nn.L;

nn.testing = 0;
%calc misclassification rate if softmax
if strcmp(nn.output,'softmax')
    [er_data, dummy]  = nntest(nn, data_x, data_y);
    loss.e_frac       = er_data;
end

end
