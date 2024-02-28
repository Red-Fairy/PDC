import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, test_dataset, eval_dataset = CreateDataset(opt)

    train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(train_dataset, shuffle=True),
            drop_last=True,
            )

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(test_dataset, shuffle=False),
            drop_last=False,
            )

    test_dl_for_eval = torch.utils.data.DataLoader(
            eval_dataset if eval_dataset is not None else test_dataset,
            batch_size=1,
            sampler=data_sampler(test_dataset, shuffle=False),
            drop_last=False,
        )

    return train_dl, test_dl, test_dl_for_eval
