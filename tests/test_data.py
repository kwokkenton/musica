from torch.utils.data import DataLoader

from muse.data import DataProcessor, MaestroDataset

dp = DataProcessor(batch_size=16)


train_ds = MaestroDataset('/data', '/data/data.csv', dp)

train_dl = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    collate_fn=dp.collate_fn,
)

for j in range(100):
    for i, batch in enumerate(train_dl):
        print(batch)
