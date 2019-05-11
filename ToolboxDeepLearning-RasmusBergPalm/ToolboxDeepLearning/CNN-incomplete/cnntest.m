function tst = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);

    tst.bad = find(h ~= a);
    tst.e_frac = numel(tst.bad) / size(y, 2);
    tst.loss = 1/2 * sum(sum(net.o - y) .^ 2) / size(y, 1);
end
