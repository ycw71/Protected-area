import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from main import FocalLoss, MultiViewGNN, prepare_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_percent(log_probs, labels):
    pred = log_probs.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100.0


def evaluate_fold(model, data_a, data_b):
    model.eval()
    with torch.no_grad():
        log_probs = model(data_a, data_b)
        probs = log_probs.exp()
        pred = probs.argmax(dim=1)
    return probs.cpu().numpy(), pred.cpu().numpy(), data_a.y.cpu().numpy()


def fold_metrics(y_true, y_pred, fold):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "fold": fold,
        "accuracy": accuracy,
        "precision_class0": precision[0],
        "recall_class0": recall[0],
        "f1_class0": f1[0],
        "support_class0": int(support[0]),
        "precision_class1": precision[1],
        "recall_class1": recall[1],
        "f1_class1": f1[1],
        "support_class1": int(support[1]),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def train_cross_validation(
    data_a,
    data_b,
    output_dir,
    epochs=300,
    n_splits=5,
    learning_rate=0.005,
    weight_decay=5e-4,
    seed=42,
    device=None,
):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = data_a.y.cpu().numpy()
    node_ids = np.asarray(data_a.node_ids)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    history_rows = []
    metric_rows = []
    prediction_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        print(f"\nFold {fold}/{n_splits}")

        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        test_idx_t = torch.tensor(test_idx, dtype=torch.long)

        train_a = data_a.subgraph(train_idx_t).to(device)
        train_b = data_b.subgraph(train_idx_t).to(device)
        test_a = data_a.subgraph(test_idx_t).to(device)
        test_b = data_b.subgraph(test_idx_t).to(device)

        model = MultiViewGNN(
            input_dim=data_a.num_features,
            embedding_dim=16,
            num_classes=2,
        ).to(device)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(train_a, train_b)
            loss = criterion(output, train_a.y)
            loss.backward()
            optimizer.step()

            train_acc = accuracy_percent(output.detach(), train_a.y)
            history_rows.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "train_accuracy": train_acc,
                }
            )

            if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"loss={loss.item():.4f} | train_acc={train_acc:.2f}%"
                )

        probs, y_pred, y_true = evaluate_fold(model, test_a, test_b)
        metrics = fold_metrics(y_true, y_pred, fold)
        metric_rows.append(metrics)

        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        for local_idx, original_idx in enumerate(test_idx):
            prediction_rows.append(
                {
                    "row_index": int(original_idx),
                    "node_id": node_ids[original_idx],
                    "fold": fold,
                    "true_label": int(y_true[local_idx]),
                    "predicted_label": int(y_pred[local_idx]),
                    "prob_class0": float(probs[local_idx, 0]),
                    "prob_class1": float(probs[local_idx, 1]),
                }
            )

        torch.save(model.state_dict(), output_dir / f"fold_{fold:02d}_model.pt")

    history_df = pd.DataFrame(history_rows)
    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows).sort_values("row_index")

    metric_columns = [
        col for col in metrics_df.columns
        if col != "fold" and not col.startswith("support_")
    ]
    summary_df = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean": [metrics_df[col].mean() for col in metric_columns],
            "std": [metrics_df[col].std(ddof=1) for col in metric_columns],
        }
    )

    history_df.to_csv(output_dir / "training_history.csv", index=False)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)
    predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    print("\nCross-validation summary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {output_dir.resolve()}")
    return metrics_df, summary_df, predictions_df, history_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-view GNN with cross-validation.")
    parser.add_argument("--data", type=Path, default=Path("./data/input_data.xlsx"))
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--output", type=Path, default=Path("./results"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    data_a, data_b, _, info = prepare_data(args.data, sheet)

    print(f"Nodes: {info['n_nodes']}")
    print(f"Spatial edges: {info['spatial_edges']}")
    print(f"Similarity edges: {info['similarity_edges']}")

    train_cross_validation(
        data_a=data_a,
        data_b=data_b,
        output_dir=args.output,
        epochs=args.epochs,
        n_splits=args.folds,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
