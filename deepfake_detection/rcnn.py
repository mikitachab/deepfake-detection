import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class RCNN(nn.Module):
    """
    stacks rnn on top of cnn
    """

    def __init__(
        self,
        num_classes=2,
        rnn_hidden_size=128,
        rnn_num_layers=2,
        cnn=None,
        n_features=None,
    ):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        if cnn is None:
            cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn = cnn
        if n_features is None:
            if self.cnn.__class__.__name__ == "VGG":
                n_features = self.cnn.classifier[0].in_features
            if self.cnn.__class__.__name__ == "ResNet":
                n_features = self.cnn.fc.in_features
        self.n_features = n_features
        if self.cnn.__class__.__name__ == "VGG":
            self.cnn.classifier = nn.Identity()
        if self.cnn.__class__.__name__ == "ResNet":
            self.cnn.fc = nn.Identity()


        self.lstm = nn.LSTM(
            n_features, rnn_hidden_size, rnn_num_layers, batch_first=True
        )

        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        """
        Note: this works only for input with batch_size=1.
        One sample is sequense of frames.
        """
        c_out = self.cnn(x)
        r_out, (h_n, h_c) = self.lstm(torch.unsqueeze(c_out, 0))
        fc_in = torch.unsqueeze(r_out[0][-1], 0)  # last hidden vector from rnn
        fc_out = self.fc(fc_in)
        out = F.softmax(fc_out, dim=1)
        return out

    def clone(self):
        return RCNN(**self.get_own_properties())

    def get_own_properties(self):
        return dict(
            num_classes=self.num_classes,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_num_layers=self.rnn_num_layers,
            cnn=self.cnn,
            n_features=self.n_features,
        )
