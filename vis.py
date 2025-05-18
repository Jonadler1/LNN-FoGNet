import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


# Define models and model-specific epochs
models = ['lstm', 'ctrnn', 'ltc']
model_epochs = {'lstm': 15, 'ctrnn': 15, 'ltc': 15}  # adjust as needed
size = 32
results_dir = 'results/fog'

##############################
# Original Visualization Code
##############################
def load_all_folds(model, size, epochs):
    """
    Load complete training history for each fold of a given model.
    Expected keys: 'loss', 'accuracy', 'precision', 'recall', 'f1_score',
                   'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score'
    """
    metric_keys = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
    val_metric_keys = ['val_' + key for key in metric_keys]
    all_keys = metric_keys + val_metric_keys

    history_metrics = {key: [] for key in all_keys}
    for fold in range(1, 6):
        history_file = os.path.join(results_dir, f"{model}_{size}_fold_{fold}_history.pkl")
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        for key in all_keys:
            if key in history:
                history_metrics[key].append(history[key])
            else:
                print(f"Warning: {key} not found in fold {fold} for model {model}.")
                history_metrics[key].append(np.zeros(epochs))
    return history_metrics

# Plot per-fold training and validation curves for each metric
for model in models:
    current_epochs = model_epochs[model]
    history_metrics = load_all_folds(model, size, current_epochs)
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1_score']:
        # Training curves
        plt.figure(figsize=(8,6))
        for fold_idx in range(len(history_metrics[metric])):
            plt.plot(range(1, current_epochs+1), history_metrics[metric][fold_idx], label=f'Fold {fold_idx+1}')
        plt.title(f'{model.upper()} Training {metric.capitalize()} by Fold')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"training_{metric}_folds_{model}.png"))
        plt.close()

        # Validation curves
        val_key = f"val_{metric}"
        plt.figure(figsize=(8,6))
        for fold_idx in range(len(history_metrics[val_key])):
            plt.plot(range(1, current_epochs+1), history_metrics[val_key][fold_idx], label=f'Fold {fold_idx+1}')
        plt.title(f'{model.upper()} Validation {metric.capitalize()} by Fold')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"validation_{metric}_folds_{model}.png"))
        plt.close()

# Plot test performance bar charts from summary CSV
test_metrics = {}
for model in models:
    summary_file = os.path.join(results_dir, f"{model}_{size}_kfold_summary.csv")
    with open(summary_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
        row = [float(val) for val in row]
        # Test metrics: indices 10-14
        test_metrics[model] = {
            'loss': row[10],
            'accuracy': row[11],
            'precision': row[12],
            'recall': row[13],
            'f1_score': row[14]
        }

for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1_score']:
    plt.figure(figsize=(8,6))
    model_names = [m.upper() for m in models]
    metric_values = [test_metrics[m][metric] for m in models]
    bars = plt.bar(model_names, metric_values, color=['skyblue','lightgreen','salmon'])
    plt.title(f'Final Test {metric.capitalize()} Comparison')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, bar in enumerate(bars):
        x = bar.get_x() + bar.get_width()/2.0
        plt.text(x, metric_values[i], f"{metric_values[i]:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.savefig(os.path.join(results_dir, f"test_{metric}_comparison.png"))
    plt.close()

##################################
# New Visualizations Section
##################################
def plot_mean_acc_loss(model, size, epochs):
    fold_histories = load_all_folds(model, size, epochs)
    x_axis = range(1, epochs+1)
    train_acc = np.mean(np.array(fold_histories['accuracy']), axis=0)
    val_acc   = np.mean(np.array(fold_histories['val_accuracy']), axis=0)
    train_loss = np.mean(np.array(fold_histories['loss']), axis=0)
    val_loss   = np.mean(np.array(fold_histories['val_loss']), axis=0)
    
    # Mean Accuracy Plot
    plt.figure(figsize=(8,6))
    plt.plot(x_axis, train_acc, label='Train Accuracy (mean)', color='blue')
    plt.plot(x_axis, val_acc, label='Validation Accuracy (mean)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model.upper()} Mean Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"mean_accuracy_{model}_{size}.png"))
    plt.close()

    # Mean Loss Plot
    plt.figure(figsize=(8,6))
    plt.plot(x_axis, train_loss, label='Train Loss (mean)', color='blue')
    plt.plot(x_axis, val_loss, label='Validation Loss (mean)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{model.upper()} Mean Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"mean_loss_{model}_{size}.png"))
    plt.close()

def plot_confusion_matrices(model, size):
    """
    Plots normalized confusion matrices for training, validation, and test sets.
    Each cell is annotated with its normalized value (percentage) and the corresponding label.
    """
    sets = ['train', 'val', 'test']
    cms = []
    for s in sets:
        cm_file = os.path.join(results_dir, f"confusion_matrix_{model}_{size}_{s}.csv")
        if not os.path.exists(cm_file):
            print(f"File not found: {cm_file}")
            cms.append(None)
        else:
            with open(cm_file, 'r') as f:
                row = next(csv.reader(f))
                cm = np.array([int(x) for x in row]).reshape(2,2)
                # Normalize each row (if row sum > 0)
                norm_cm = np.zeros_like(cm, dtype=float)
                for i in range(2):
                    row_sum = np.sum(cm[i])
                    if row_sum > 0:
                        norm_cm[i] = cm[i] / row_sum
                cms.append(norm_cm)
    cell_labels = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    set_titles = ['Training', 'Validation', 'Test']
    for idx, cm in enumerate(cms):
        ax = axes[idx]
        if cm is None:
            ax.set_title(f"{set_titles[idx]}: No data")
            ax.axis('off')
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{set_titles[idx]}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['No Fog', 'Fog'])
            ax.set_yticklabels(['No Fog', 'Fog'])
            thresh = cm.max() / 2.
            for i in range(2):
                for j in range(2):
                    # Display normalized value as percentage
                    text = f"{cell_labels[(i,j)]}: {cm[i,j]*100:.1f}%"
                    ax.text(j, i, text, ha="center", va="center",
                            color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrices_{model}_{size}.png"))
    plt.close()

def plot_roc_pr_folds(model, size):
    """
    Loads per-fold ROC/PR data from a pickle file and plots:
      - ROC curves for each fold (continuous lines) and a dashed reference diagonal line.
      - PR curves for each fold (continuous lines) and a dashed horizontal baseline for no-skill.
      - The mean ROC and PR curves (solid black) with corresponding AUC/AUPR.
    """
    rocpr_file = os.path.join(results_dir, f"{model}_{size}_fold_rocpr.pkl")
    if not os.path.exists(rocpr_file):
        print(f"No fold ROC/PR data found: {rocpr_file}")
        return
    with open(rocpr_file, 'rb') as f:
        data = pickle.load(f)
    fold_roc = data['fold_roc_test']
    fold_pr = data['fold_pr_test']

    # ROC Plot
    plt.figure(figsize=(6,6))
    all_fpr = np.linspace(0,1,1000)
    tprs = []
    aucs = []
    for fold, (fpr, tpr) in fold_roc.items():
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC={fold_auc:.2f})')  # continuous line
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot([0,1], [0,1], '--', color='gray', label="No Skill (Random)")
    plt.plot(all_fpr, mean_tpr, 'k-', lw=2, label=f'Mean ROC (AUC={mean_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model.upper()} ROC Curves (Folds)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"roc_folds_{model}_{size}.png"))
    plt.close()

    # PR Plot
    plt.figure(figsize=(6,6))
    all_rec = np.linspace(0,1,1000)
    precs = []
    aps = []
    for fold, (precision, recall) in fold_pr.items():
        idx_sort = np.argsort(recall)
        rec_sorted = np.array(recall)[idx_sort]
        prec_sorted = np.array(precision)[idx_sort]
        area_pr = np.trapezoid(prec_sorted, rec_sorted)
        aps.append(area_pr)
        plt.plot(rec_sorted, prec_sorted, label=f'Fold {fold+1} (AUPR={area_pr:.2f})')  # continuous line
        interp_prec = np.interp(all_rec, rec_sorted, prec_sorted, left=prec_sorted[0], right=prec_sorted[-1])
        precs.append(interp_prec)
    mean_prec = np.mean(precs, axis=0)
    mean_ap = np.trapezoid(mean_prec, all_rec)
    # For PR, add a dashed horizontal baseline at the no-skill level.
    # No-skill level is the ratio of positive samples in the test set.
    test_cm_file = os.path.join(results_dir, f"confusion_matrix_{model}_{size}_test.csv")
    if os.path.exists(test_cm_file):
        with open(test_cm_file, 'r') as f:
            row = next(csv.reader(f))
            cm_vals = [int(x) for x in row]
            cm = np.array(cm_vals).reshape(2,2)
            total = cm.sum()
            pos_count = cm[1,0] + cm[1,1]
            baseline_level = pos_count / total if total > 0 else 0.0
    else:
        baseline_level = 0.0
   #plt.plot([0,1], [baseline_level, baseline_level], '--', color='gray', label=f'No Skill (baseline={baseline_level:.2f})')
    plt.plot(all_rec, mean_prec, 'k-', lw=2, label=f'Mean PR (AUPR={mean_ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model.upper()} PR Curves (Folds)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"pr_folds_{model}_{size}.png"))
    plt.close()
    
    
###########################################
#  NEW: composite ROC & PR panel figures  #
###########################################

def plot_roc_pr_panels(models, size, results_dir):
    """
    Build two composite figures:
      • roc_panels_{size}.png  – 3-panel ROC (LSTM, CTRNN, LTC)
      • pr_panels_{size}.png   – 3-panel PR  (LSTM, CTRNN, LTC)
    Each panel is identical to the per-model figure you already make, 
    but they sit together in one file.
    """
    # Helper that plots *into* a supplied axis instead of creating a new fig
    def _plot_single(ax, model, metric):
        rocpr_file = os.path.join(results_dir, f"{model}_{size}_fold_rocpr.pkl")
        with open(rocpr_file, "rb") as f:
            data = pickle.load(f)

        fold_curves = data[f"fold_{metric}_test"]          # dict {fold: (x,y)}
        all_x = np.linspace(0, 1, 1000)
        ys, areas = [], []

        for fold, (y1, y2) in fold_curves.items():
            # metric == 'roc'  ➜ (fpr, tpr)
            # metric == 'pr'   ➜ (precision, recall)  → sort by recall
            if metric == 'pr':
                idx = np.argsort(y2)        # y2 = recall
                x, y = np.array(y2)[idx], np.array(y1)[idx]
                area = np.trapezoid(y, x)
            else:                           # ROC
                x, y = y1, y2               # x=fpr, y=tpr
                area = auc(x, y)

            lbl = f"Fold {fold+1} ({'AUC' if metric=='roc' else 'AUPR'}={area:.2f})"
            ax.plot(x, y, lw=1, label=lbl)
            ys.append(np.interp(all_x, x, y))
            areas.append(area)

        # mean curve
        mean_y = np.mean(ys, axis=0)
        mean_area = np.mean(areas)
        ax.plot(all_x, mean_y, 'k-', lw=2,
                label=f"Mean {'ROC' if metric=='roc' else 'PR'}"
                      f" ({'AUC' if metric=='roc' else 'AUPR'}={mean_area:.2f})")

        # reference lines
        if metric == 'roc':
            ax.plot([0, 1], [0, 1], '--', color='gray', label='No Skill')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
        else:   # PR
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim(0.75, 1.00)

        ax.set_title(model.upper())
        ax.grid(True)
        ax.legend(fontsize=7, loc="lower right")

    # ---------- build ROC and PR figures ----------
    for metric in ("roc", "pr"):
        fig = plt.figure(figsize=(6, 10))
        gs  = GridSpec(3, 1, hspace=0.35)
        for i, model in enumerate(models):
            ax = fig.add_subplot(gs[i, 0])
            _plot_single(ax, model, metric)
            ax.text(-0.12, 1.05, chr(65+i), transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')   # Panel label A/B/C

        fig.suptitle(f"{metric.upper()} curves – 5-fold cross-validation",
                     fontsize=14, y=0.93)
        outname = os.path.join(results_dir, f"{metric}_panels_{size}.png")
        fig.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
# ---------------------------------------------------------
# NEW FUNCTIONS: combined convergence plots (train / val)
# ---------------------------------------------------------
def compute_mean_accuracy(history_dict, metric_key, epochs):
    """Return a 1-D numpy array of mean accuracy across folds."""
    return np.mean(np.array(history_dict[metric_key])[:, :epochs], axis=0)

def plot_combined_accuracy(models, size, model_epochs, results_dir, which="train"):
    """
    which = "train"  ➜ use 'accuracy'
    which = "val"    ➜ use 'val_accuracy'
    Produces one figure with three colour-coded lines.
    """
    plt.figure(figsize=(8, 6))

    colours = {"lstm": "royalblue",
               "ctrnn": "seagreen",
               "ltc":   "tomato"}

    for model in models:
        epochs = model_epochs[model]
        hist   = load_all_folds(model, size, epochs)
        key    = "accuracy" if which == "train" else "val_accuracy"
        mean_acc = compute_mean_accuracy(hist, key, epochs)
        plt.plot(range(1, epochs + 1),
                 mean_acc,
                 label=f"{model.upper()}  (final={mean_acc[-1]:.3f})",
                 color=colours[model],
                 linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    title_stub = "Training" if which == "train" else "Validation"
    plt.title(f"Mean {title_stub} Accuracy – Convergence Comparison")
    plt.legend()
    plt.grid(True)
    outname = f"combined_{which}_accuracy.png"
    plt.savefig(os.path.join(results_dir, outname), dpi=300, bbox_inches="tight")
    plt.close()


##############################
# Main Section: Call Visualizations
##############################
if __name__ == "__main__":
    # Original visualizations (per-fold training/validation curves and test bar charts) are generated first.
    for model in models:
        print(f"Generating original visualizations for {model.upper()}")
    # New visualizations:
    for model in models:
        current_epochs = model_epochs[model]
        print(f"Generating new visualizations for {model.upper()} using {current_epochs} epochs")
        plot_mean_acc_loss(model, size, current_epochs)
        plot_confusion_matrices(model, size)
        plot_roc_pr_folds(model, size)
        plot_roc_pr_panels(models, size, results_dir)
        plot_combined_accuracy(models, size, model_epochs, results_dir, which="train")
        plot_combined_accuracy(models, size, model_epochs, results_dir, which="val")
    print("All visualizations generated.")
