"""
datasets.py — Setup for Waterbirds, CelebA, CivilComments, MultiNLI
=====================================================================

Every setup function returns the same 7 things:
    train_loader, val_loader, test_loader, model, optimizer, scheduler, config

Batch format:
    Vision  — (x_tensor, y_tensor, metadata_tensor)  from WILDS
    NLP     — (x_dict,   y_tensor, metadata_tensor)  custom collate
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────
# 1. WATERBIRDS
# ─────────────────────────────────────────────────────────────────

def setup_waterbirds(data_dir="./data"):
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader

    tf_train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = get_dataset(dataset="waterbirds", download=True, root_dir=data_dir)
    tr = get_train_loader("standard", ds.get_subset("train", transform=tf_train),
                          batch_size=128, num_workers=4)
    va = get_eval_loader("standard", ds.get_subset("val",   transform=tf_eval),
                         batch_size=256, num_workers=4)
    te = get_eval_loader("standard", ds.get_subset("test",  transform=tf_eval),
                         batch_size=256, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, ds.n_classes)

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    cfg = dict(dataset="waterbirds", n_classes=ds.n_classes, epochs=100,
               lr=1e-3, is_nlp=False,
               groups={0: "landbird+land", 1: "landbird+water",
                       2: "waterbird+land", 3: "waterbird+water"})
    return tr, va, te, model, opt, sch, cfg


# ─────────────────────────────────────────────────────────────────
# 2. CELEBA
# ─────────────────────────────────────────────────────────────────

def setup_celeba(data_dir="./data"):
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader

    tf_train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = get_dataset(dataset="celebA", download=True, root_dir=data_dir)
    tr = get_train_loader("standard", ds.get_subset("train", transform=tf_train),
                          batch_size=128, num_workers=4)
    va = get_eval_loader("standard", ds.get_subset("val",   transform=tf_eval),
                         batch_size=256, num_workers=4)
    te = get_eval_loader("standard", ds.get_subset("test",  transform=tf_eval),
                         batch_size=256, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, ds.n_classes)

    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    cfg = dict(dataset="celeba", n_classes=ds.n_classes, epochs=50,
               lr=1e-4, is_nlp=False,
               groups={0: "dark+male", 1: "dark+female",
                       2: "blond+male", 3: "blond+female"})
    return tr, va, te, model, opt, sch, cfg


# ─────────────────────────────────────────────────────────────────
# 3. CIVILCOMMENTS
# ─────────────────────────────────────────────────────────────────

def _cc_collate(batch):
    """Custom collate for CivilComments: produces (x_dict, y, metadata)."""
    xs, ys, ms = [], [], []
    for x, y, m in batch:
        xs.append(x)
        ys.append(y)
        ms.append(m)
    # x is a dict of tensors after tokenization
    x_batch = {k: torch.stack([xi[k] for xi in xs]) for k in xs[0].keys()}
    y_batch = torch.stack(ys) if isinstance(ys[0], torch.Tensor) else torch.tensor(ys)
    m_batch = torch.stack(ms) if isinstance(ms[0], torch.Tensor) else torch.tensor(ms)
    return x_batch, y_batch, m_batch


def setup_civilcomments(data_dir="./data"):
    from wilds import get_dataset
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(text):
        t = tokenizer(str(text), padding="max_length", truncation=True,
                      max_length=300, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in t.items()}

    ds = get_dataset(dataset="civilcomments", download=True, root_dir=data_dir)

    tr = torch.utils.data.DataLoader(
        ds.get_subset("train", transform=tokenize),
        batch_size=16, shuffle=True, num_workers=2, collate_fn=_cc_collate)
    va = torch.utils.data.DataLoader(
        ds.get_subset("val", transform=tokenize),
        batch_size=32, shuffle=False, num_workers=2, collate_fn=_cc_collate)
    te = torch.utils.data.DataLoader(
        ds.get_subset("test", transform=tokenize),
        batch_size=32, shuffle=False, num_workers=2, collate_fn=_cc_collate)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=ds.n_classes)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                             total_iters=5)

    cfg = dict(dataset="civilcomments", n_classes=ds.n_classes, epochs=5,
               lr=1e-5, is_nlp=True)
    return tr, va, te, model, opt, sch, cfg


# ─────────────────────────────────────────────────────────────────
# 4. MULTINLI
# ─────────────────────────────────────────────────────────────────

_NEG = {"no","not","never","nothing","nobody","none","neither","nor",
        "nowhere","cannot","can't","won't","don't","doesn't","didn't",
        "isn't","aren't","wasn't","weren't"}


class _MultiNLI(torch.utils.data.Dataset):
    """Wraps HuggingFace MultiNLI into (x_dict, y, metadata) format."""
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.data = hf_dataset
        self.tok = tokenizer
        self.ml = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        enc = self.tok(row["premise"], row["hypothesis"],
                       padding="max_length", truncation=True,
                       max_length=self.ml, return_tensors="pt")
        x = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(row["label"], dtype=torch.long)
        neg = 1 if len(set(row["hypothesis"].lower().split()) & _NEG) > 0 else 0
        group = row["label"] * 2 + neg
        meta = torch.tensor([group], dtype=torch.long)
        return x, y, meta


def _mnli_collate(batch):
    xs, ys, ms = zip(*batch)
    x_batch = {k: torch.stack([xi[k] for xi in xs]) for k in xs[0].keys()}
    return x_batch, torch.stack(ys), torch.stack(ms)


def setup_multinli(data_dir="./data"):
    from datasets import load_dataset
    from transformers import BertTokenizer, BertForSequenceClassification

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    raw = load_dataset("multi_nli")

    tr_ds = _MultiNLI(raw["train"], tokenizer)
    va_ds = _MultiNLI(raw["validation_matched"], tokenizer)

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True,
                                      num_workers=2, collate_fn=_mnli_collate)
    va = torch.utils.data.DataLoader(va_ds, batch_size=64, shuffle=False,
                                      num_workers=2, collate_fn=_mnli_collate)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                             total_iters=3)

    cfg = dict(dataset="multinli", n_classes=3, epochs=3,
               lr=2e-5, is_nlp=True,
               groups={0:"entail+no_neg", 1:"entail+neg",
                       2:"neutral+no_neg", 3:"neutral+neg",
                       4:"contra+no_neg", 5:"contra+neg"})
    return tr, va, va, model, opt, sch, cfg  # no separate test split
