clear all

Images = loadMNISTImages('F:\Workspace\Matlab\Deeplearning\FirstDeepLearning\MNIST_data\t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);

Labels = loadMNISTLabels('F:\Workspace\Matlab\Deeplearning\FirstDeepLearning\MNIST_data\t10k-labels.idx1-ubyte');
Labels(Labels==0)=10;

X = Images(:, :, 1:8000);
D = Labels(1:8000);

W1 = 0.01 * randn([9 9 20]);
W5 = 2 * rand(100, 2000) -1;
W0 = 2 * rand(10, 100) -1;

for epoch = 1:3
    epoch
    [W1 W5 W0] = MnistConv(W1, W5, W0, X, D);
end
%save('MnistConv.mat');
X = Images(:,:,8001:10000);
D = Labels(8001:10000);
acc = 0;
N = length(D);
for k = 1:N
    x = X(:, :, k);
    y1 = Conv(x, W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4 = reshape(y3, [], 1);
    v5 = W5 * y4;
    y5 = ReLU(v5);
    v6 = W0 * y5;
    y6 = Softmax(v6);
    
    [~, i] = max(y6);
    if i == D(k)
        acc = acc+1;
    end
end
acc = acc/N;
fprintf('Accuracy is %f\n', acc);