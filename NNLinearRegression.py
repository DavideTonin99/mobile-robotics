from torch import nn


class NNLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Linear Regression Model
        :param input_dim: input dimension
        :param output_dim: output dimension
        """
        super(NNLinearRegression, self).__init__()
        self.bn_cont = nn.BatchNorm1d(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=output_dim, bias=True)
        )

    def forward(self, x):
        x = self.bn_cont(x)
        out = self.layers(x)
        return out
