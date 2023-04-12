"""
Create dataloader for pre-training
"""

__all__ = [
    "create_loader_pretrain"
]

def create_loader_pretrain(
    dataset,
    batch_size,
    drop_remainder=False,
    transform=None,
    num_parallel_workers=None,
    python_multiprocessing=False
):
    if transform is None:
        raise ValueError("tranform should not be None for pre-training.")

    dataset = dataset.map(
        operations=transform,
        input_columns="image",
        output_columns=transform.output_columns,
        column_order=transform.output_columns,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing
    )

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return dataset
