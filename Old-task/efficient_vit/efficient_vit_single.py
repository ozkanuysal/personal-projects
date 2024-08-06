import torch
import torch.nn as nn
import timm


def create_model():
    model = EfficientVit(in_channels=3, out_channels=5)

    for name, params in model.named_parameters():
        if 'classifier' in name: # Make sure there are no classifier layers among previous layers 
            print(f"Layer requiring gradient: {name}")
            continue
        params.requires_grad = False
        #print(f"Layer frozen: {name}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
    print(count_trainable_parameters(model))

    return model

class EfficientVit(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, pretrained=False):
        super(EfficientVit, self).__init__()

        self.model = timm.create_model(
            'efficientvit_b3',
            pretrained=pretrained,
        )

        if in_channels != 3:
            self.model.stem.in_conv.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.head.classifier[4] = nn.Linear(in_features=2560, out_features=out_channels, bias=True)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    import os
    import numpy as np
    from time import perf_counter

    model = create_model().to(torch.device('cuda'))
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    BN = 8
    H, W = 512, 512
    EPOCH = 20
    average_time = []

    for i in range(EPOCH + 2):
        start_time = perf_counter()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y_true = torch.randn(BN if i != 0 else 1, 3, H, W).to(torch.device('cuda')), torch.randint(0, 5, (BN if i != 0 else 1,)).to(torch.device('cuda'))
            y = model(x)
            loss = nn.functional.cross_entropy(y, y_true)

        loss.backward()
        if i == 1:
            print(os.system('nvidia-smi'))
        optimizer.step()
        
        if i > 1:
            stop_time = perf_counter() - start_time
            average_time.append(stop_time)
            print(f"Elapsed Time: {stop_time :.5f} sec. (BN = {BN})")

    print(f"Average Elapsed Time: {np.mean(average_time) :.5f} sec.")