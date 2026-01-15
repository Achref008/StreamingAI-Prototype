import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from config import NODE_ID, DIRICHLET_ALPHA, PEER_VALIDATION_MAP, DATASET_TYPE_MAP, AKIDA_CIFAR_RATIO
from sklearn.model_selection import StratifiedShuffleSplit


# -------------------------------------------------------------------------
# Validation loader: local validation or cross-node (peer) validation
# -------------------------------------------------------------------------
def get_validation_loader(peer=True):
    """
    Builds a reproducible validation loader.

    - peer=True: validate on the dataset associated with the mapped peer node
      (cross-node evaluation to measure generalization under heterogeneity).
    - peer=False: validate on the local node’s dataset.
    """
    target_node = PEER_VALIDATION_MAP[NODE_ID] if peer else NODE_ID
    dataset_type = DATASET_TYPE_MAP[target_node]

    # Node 7 uses 1-channel (grayscale) 28x28 inputs.
    # When validating on CIFAR10, convert CIFAR RGB -> grayscale and resize to 28x28
    # to match the Akida-side model input constraints.
    if NODE_ID == 7 and dataset_type == "CIFAR10":
        validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        validation_transform = get_transform(dataset_type)

    test_dataset = load_dataset(dataset_type, train=False, transform=validation_transform)

    # Fixed validation subset for reproducibility across runs
    np.random.seed(1337 + target_node)

    # Robust label extraction to support stratified sampling
    if hasattr(test_dataset, 'targets'):
        targets = np.array(test_dataset.targets)
    elif hasattr(test_dataset, 'labels'):
        targets = np.array(test_dataset.labels)
    else:
        try:
            targets = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
        except Exception as e:
            print(f"Warning: Could not extract targets for stratified validation. Error: {e}")
            targets = np.arange(len(test_dataset))

    # Stratified sampling keeps the class proportions stable in the validation subset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2000, random_state=1337 + target_node)
    _, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    subset = Subset(test_dataset, val_idx)

    # Node 7 uses conservative DataLoader settings to avoid memory pressure
    num_workers_val = 0 if NODE_ID == 7 else 2
    pin_memory_val = False if NODE_ID == 7 else True

    return DataLoader(
        subset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=pin_memory_val
    )


# -------------------------------------------------------------------------
# Dataset factory: returns torchvision datasets with given transform
# -------------------------------------------------------------------------
def load_dataset(dataset_type, train=True, transform=None):
    root = './data'
    if dataset_type == "MNIST":
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    elif dataset_type == "CIFAR10":
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")


# -------------------------------------------------------------------------
# Warm-up loader: MNIST pretraining / initial stabilization routine
# -------------------------------------------------------------------------
def get_mnist_loader_for_warmup():
    """
    Returns an MNIST DataLoader for warm-up routines.
    Uses standard MNIST normalization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)


# -------------------------------------------------------------------------
# Transform selection: ensures consistent preprocessing per dataset/node type
# -------------------------------------------------------------------------
def get_transform(dataset_type):
    if dataset_type == "MNIST":
        # MNIST normalization (used by Node 7 and kept consistent if other nodes ever load MNIST)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == "CIFAR10":
        # CIFAR10 preprocessing with mild augmentation for training robustness
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616))
        ])

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# -------------------------------------------------------------------------
# Non-IID Dirichlet split: assigns CIFAR10 indices to multiple clients
# -------------------------------------------------------------------------
def dirichlet_split_noniid(dataset, n_clients, alpha):
    """
    Generates a non-IID split using a Dirichlet distribution over classes.

    For each class, samples client proportions ~ Dirichlet(alpha),
    then distributes that class’ indices accordingly.
    """
    n_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    idx_by_class = {i: np.where(labels == i)[0] for i in range(n_classes)}

    np.random.seed(42)
    proportions = np.random.dirichlet(np.repeat(alpha, n_clients), n_classes)

    client_indices = [[] for _ in range(n_clients)]

    for c, fracs in enumerate(proportions):
        class_indices = idx_by_class[c]
        np.random.shuffle(class_indices)

        split_points = np.cumsum(fracs) * len(class_indices)
        split_points = np.round(split_points).astype(int)[:-1]
        split_indices = np.split(class_indices, split_points)

        for i, indices in enumerate(split_indices):
            if len(indices) > 0:
                client_indices[i].extend(indices.tolist())

    return {i: np.array(indices) for i, indices in enumerate(client_indices)}


# -------------------------------------------------------------------------
# Training data loader: Node 7 hybrid dataset, others CIFAR10 non-IID
# -------------------------------------------------------------------------
def get_data(batch_size=128):
    """
    Returns the local training DataLoader depending on node type:

    - Node 7 (Akida): hybrid dataset (MNIST + subset of CIFAR10 converted to grayscale/28x28)
    - Other nodes (Jetsons): CIFAR10 split via Dirichlet non-IID assignment
    """
    if NODE_ID == 7:
        # Node 7 uses smaller batches and grayscale 28x28 inputs
        batch_size = min(batch_size, 64)

        mnist_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        cifar_node7_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
        cifar_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=cifar_node7_transform)

        # CIFAR subset size is derived from AKIDA_CIFAR_RATIO with a safety margin
        total_cifar_subset_size = int(AKIDA_CIFAR_RATIO * 1.2 * len(cifar_dataset))
        total_cifar_subset_size = min(total_cifar_subset_size, len(cifar_dataset))

        np.random.seed(NODE_ID)
        cifar_indices = np.random.choice(
            len(cifar_dataset),
            size=total_cifar_subset_size,
            replace=False
        )

        combined_dataset = ConcatDataset([
            mnist_dataset,
            Subset(cifar_dataset, cifar_indices)
        ])

        print(f"[Node 7] Loaded {len(mnist_dataset)} MNIST + {total_cifar_subset_size} CIFAR10 grayscale samples.")

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    else:
        # CIFAR10 client nodes use a Dirichlet non-IID split for local training data
        dataset_type = "CIFAR10"
        transform = get_transform(dataset_type)
        dataset = load_dataset(dataset_type, train=True, transform=transform)

        cifar_nodes = sorted([nid for nid, dtype in DATASET_TYPE_MAP.items() if dtype == "CIFAR10"])
        if NODE_ID not in cifar_nodes:
            raise ValueError(f"Node {NODE_ID} is not part of CIFAR10 client list.")

        node_id_to_client_idx = {nid: idx for idx, nid in enumerate(cifar_nodes)}
        local_id = node_id_to_client_idx[NODE_ID]
        n_clients = len(cifar_nodes)

        np.random.seed(42)
        client_map = dirichlet_split_noniid(dataset, n_clients=n_clients, alpha=DIRICHLET_ALPHA)
        indices = client_map.get(local_id, [])

        if len(indices) == 0:
            raise ValueError(
                f"No samples assigned to Node {NODE_ID} (client index {local_id}). "
                f"This can happen with very low DIRICHLET_ALPHA or few total samples."
            )

        print(f"[Node {NODE_ID}] Loaded {len(indices)} samples of {dataset_type}.")
        subset = Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
