import os
import csv
import pickle
from collections import defaultdict
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from ctrnn_model import CTRNN, NODE, CTGRU
import ltc_model as ltc

# Additional imports for plotting ROC/PR curves fold-by-fold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


def load_fog_data(folder_path):
    all_data = defaultdict(list)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            subject_id = file_name[:3]  # e.g. 'S01'
            with open(file_path, 'r') as file:
                for line in file:
                    values = line.strip().split()
                    values = [float(v) for v in values]
                    all_data[subject_id].append(values)
    for subject_id in all_data.keys():
        all_data[subject_id] = np.array(all_data[subject_id], dtype=np.float32)
    return all_data

def compute_normalization_stats(data):
    # Compute mean (and std if needed) from columns 1..-2 (excluding time=col0, annotation=last col)
    feature_columns = data[:, 1:-1]
    means = np.mean(feature_columns, axis=0)
    return means

def apply_demean_normalization(data, means):
    feature_cols = data[:, 1:-1]
    demeaned_features = feature_cols - means
    demeaned_data = np.hstack((data[:, [0]], demeaned_features, data[:, [-1]]))
    return demeaned_data

def upsample_window(window, target_size):
    """
    Upsamples a small window to 'target_size' using linear interpolation.
    window shape: (small_window_size, n_features)
    """
    current_size = window.shape[0]
    n_features = window.shape[1]
    upsampled = np.zeros((target_size, n_features))
    x_old = np.linspace(0, 1, current_size)
    x_new = np.linspace(0, 1, target_size)
    for f in range(n_features):
        upsampled[:, f] = np.interp(x_new, x_old, window[:, f])
    return upsampled

def create_windows(data, window_size=64, overlap=32, extended_factor=1.25, 
                   small_window_size=7, small_window_step=1):
    """
    Splits the data into windows. For fog windows (>40% of samples with label=2), 
    we also generate micro-segmented windows upsampled to 'window_size'.
    """
    sequences_x = []
    sequences_y = []
    step = window_size - overlap
    total_length = len(data)

    for i in range(0, total_length - window_size + 1, step):
        fixed_window = data[i:i+window_size]
        window_annotations = fixed_window[:, -1]
        # Skip if annotation=0 anywhere in the window
        if np.any(window_annotations == 0):
            continue
        fog_count = np.sum(window_annotations == 2)
        label = 1.0 if fog_count > 0.4 * window_size else 0.0
        # Always append the fixed window
        sequences_x.append(fixed_window[:, 1:-1])
        sequences_y.append(label)

        if label == 1.0:
            # For a fog window, do micro-segmentation
            extended_window_size = int(window_size * extended_factor)
            if i + extended_window_size <= total_length:
                source_window = data[i:i+extended_window_size]
            else:
                source_window = fixed_window

            src_len = len(source_window)
            for j in range(0, src_len - small_window_size + 1, small_window_step):
                candidate = source_window[j:j+small_window_size]
                if np.any(candidate[:, -1] == 0):
                    continue
                fog_count_candidate = np.sum(candidate[:, -1] == 2)
                if fog_count_candidate > 0.4 * small_window_size:
                    # Upsample
                    upsampled = upsample_window(candidate[:, 1:-1], window_size)
                    sequences_x.append(upsampled)
                    sequences_y.append(1.0)

    sequences_x = np.array(sequences_x, dtype=np.float32)
    sequences_y = np.array(sequences_y, dtype=np.float32)
    return sequences_x, sequences_y


class GaitModel:
    def __init__(self, model_type, model_size, input_shape,
                 learning_rate=0.001, dropout_rate=0.5, l2_reg=1e-4):
        self.model_type = model_type
        self.input_shape = input_shape
        self.model_size = model_size

        self.x = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='x')
        head = self.x
        if model_type == "lstm":
            self.rnn_cell = tf.keras.layers.LSTM(
                units=model_size, 
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate
            )
            head = self.rnn_cell(head)
        elif model_type.startswith("ltc"):
            learning_rate = 0.0001
            from ltc_model import LTCCell
            self.rnn_cell = LTCCell(model_size)
            head = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True)(head)
            head = tf.keras.layers.Dropout(dropout_rate)(head)
        elif model_type == "ctrnn":
            self.rnn_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True)(head)
            head = tf.keras.layers.Dropout(dropout_rate)(head)
        else:
            raise ValueError(f"Unknown model type '{model_type}'")

        head = tf.keras.layers.GlobalAveragePooling1D()(head)
        self.y = tf.keras.layers.Dense(1, activation='sigmoid',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(head)

        self.model = tf.keras.Model(inputs=self.x, outputs=self.y)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     F1Score(name='f1_score')]
        )

    def fit(self, train_x, train_y, val_x, val_y, epochs, batch_size=64):
        history = self.model.fit(
            x=train_x,
            y=train_y,
            validation_data=(val_x, val_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        # LTC param constraints if used
        if isinstance(self.rnn_cell, ltc.LTCCell):
            constrain_ops = self.rnn_cell.get_param_constrain_op()
            for op in constrain_ops:
                if isinstance(op, tf.Variable):
                    op.assign(tf.clip_by_value(op, op.numpy().min(), op.numpy().max()))

        history.best_epoch = epochs
        return history

# Helper function to store epoch metrics so we can compute mean training/validation curves
def store_fold_metrics(history, fold_idx, fold_metrics):
    # fold_metrics is a dict that accumulates arrays for each metric across folds
    # We'll store them in shape (n_folds, n_epochs)
    for key in ['loss', 'accuracy', 'precision', 'recall', 'f1_score']:
        if key in history:
            fold_metrics[key].append(history[key])
        else:
            # In case a metric is missing
            fold_metrics[key].append([0]*len(history['loss']))

    for key in ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score']:
        if key in history:
            fold_metrics[key].append(history[key])
        else:
            fold_metrics[key].append([0]*len(history['loss']))

def evaluate_and_save_curves_for_fold(fold_idx, model, x_data, y_data, 
                                      fold_roc, fold_pr):
    """
    Compute ROC and PR for each fold individually, storing TPR/FPR (and precision/recall) 
    so we can plot them later in vis.py.
    """
    preds = model.predict(x_data).flatten()
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_data, preds)
    fold_roc[fold_idx] = (fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_data, preds)
    fold_pr[fold_idx] = (precision, recall)

def save_fold_arrays(fold_roc, fold_pr, file_path):
    """
    Save the per-fold ROC and PR data to a pickle, so vis.py can load and plot them 
    with colored lines for each fold and compute mean curves.
    """
    data = {
        'fold_roc': fold_roc,
        'fold_pr': fold_pr
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="ctrnn", type=str, help="Type of model: lstm | ctrnn | ltc")
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs for training")
    parser.add_argument('--size', default=32, type=int, help="Number of units in the RNN cell")
    parser.add_argument('--k', default=5, type=int, help="Number of folds for K-Fold CV")
    args = parser.parse_args()

    results_dir = 'results/fog'
    os.makedirs(results_dir, exist_ok=True)

    all_data = load_fog_data(folder_path='fog_data')
    # Remove subjects with no freezing events if needed
    subjects_to_exclude = ['S04', 'S10']
    for subj in subjects_to_exclude:
        if subj in all_data:
            del all_data[subj]

    subjects = list(all_data.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.k, shuffle=False)

    # We'll accumulate fold metrics to compute mean training/validation curves
    fold_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1_score': []
    }

    # For storing fold-by-fold ROC/PR data for the test set
    fold_roc_test = {}
    fold_pr_test = {}

    # Also store confusion matrix accumulations for training, validation, test
    # to create combined confusion matrices
    train_true_all, train_pred_all = [], []
    val_true_all, val_pred_all = [], []
    test_true_all, test_pred_all = [], []

    fold_results = []

    for fold_idx, (train_val_index, test_index) in enumerate(kf.split(subjects)):
        print(f"Starting fold {fold_idx+1}/{args.k}")
        test_subjects = [subjects[i] for i in test_index]
        train_val_subjects = [subjects[i] for i in train_val_index]

        # Let's define 20% of the train_val_subjects as validation
        val_ratio = 0.2
        val_count = int(len(train_val_subjects) * val_ratio)
        np.random.shuffle(train_val_subjects)
        val_subjects = train_val_subjects[:val_count]
        train_subjects = train_val_subjects[val_count:]

        def combine_subject_data(subj_list):
            combined = []
            for s in subj_list:
                combined.append(all_data[s])
            if len(combined) == 0:
                return None
            return np.concatenate(combined, axis=0)

        train_data = combine_subject_data(train_subjects)
        val_data = combine_subject_data(val_subjects)
        test_data = combine_subject_data(test_subjects)

        # Demean
        train_means = compute_normalization_stats(train_data)
        train_data_norm = apply_demean_normalization(train_data, train_means)
        val_data_norm = apply_demean_normalization(val_data, train_means)
        test_data_norm = apply_demean_normalization(test_data, train_means)

        # Create windows
        train_x, train_y = create_windows(train_data_norm, window_size=64, overlap=32, extended_factor=1.25,
                                          small_window_size=7, small_window_step=1)
        val_x, val_y = create_windows(val_data_norm, window_size=64, overlap=32, extended_factor=1.25,
                                      small_window_size=7, small_window_step=1)
        test_x, test_y = create_windows(test_data_norm, window_size=64, overlap=32)

        if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
            print(f"Skipping fold {fold_idx+1} due to no valid windows.")
            continue

        # Build model
        gait_model = GaitModel(model_type=args.model, model_size=args.size, input_shape=(64, 9))
        history = gait_model.fit(train_x, train_y, val_x, val_y, epochs=args.epochs, batch_size=64)

        # Save training history for plotting
        history_file = f"{results_dir}/{args.model}_{args.size}_fold_{fold_idx+1}_history.pkl"
        with open(history_file, 'wb') as pkl_file:
            pickle.dump(history.history, pkl_file)

        # Accumulate epoch metrics
        store_fold_metrics(history.history, fold_idx, fold_metrics)

        # Evaluate on training, validation, test
        train_eval = gait_model.model.evaluate(train_x, train_y, verbose=0)
        val_eval = gait_model.model.evaluate(val_x, val_y, verbose=0)
        test_eval = gait_model.model.evaluate(test_x, test_y, verbose=0)

        # train_eval = [loss, acc, precision, recall, f1]
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_eval
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_eval
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_eval

        print(f"Fold {fold_idx+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
              f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        fold_results.append((train_loss, train_acc, train_precision, train_recall, train_f1,
                             val_loss, val_acc, val_precision, val_recall, val_f1,
                             test_loss, test_acc, test_precision, test_recall, test_f1))

        # Store predictions for confusion matrix
        train_preds = (gait_model.model.predict(train_x) >= 0.5).astype(np.float32).flatten()
        val_preds   = (gait_model.model.predict(val_x)   >= 0.5).astype(np.float32).flatten()
        test_preds  = (gait_model.model.predict(test_x)  >= 0.5).astype(np.float32).flatten()
        train_true_all.extend(train_y.tolist())
        train_pred_all.extend(train_preds.tolist())
        val_true_all.extend(val_y.tolist())
        val_pred_all.extend(val_preds.tolist())
        test_true_all.extend(test_y.tolist())
        test_pred_all.extend(test_preds.tolist())

        # Compute per-fold ROC/PR for the test set
        # (storing so we can later plot fold lines plus the mean in vis.py)
        from sklearn.metrics import roc_curve, precision_recall_curve
        test_prob = gait_model.model.predict(test_x).flatten()
        fpr, tpr, _ = roc_curve(test_y, test_prob)
        precision, recall, _ = precision_recall_curve(test_y, test_prob)
        # We'll save these in a pickle after the loop

        fold_roc_test[fold_idx] = (fpr, tpr)
        fold_pr_test[fold_idx]  = (precision, recall)

    # Compute aggregated confusion matrices for train/val/test
    cm_train = confusion_matrix(train_true_all, train_pred_all, labels=[0,1])
    cm_val   = confusion_matrix(val_true_all,   val_pred_all,   labels=[0,1])
    cm_test  = confusion_matrix(test_true_all,  test_pred_all,  labels=[0,1])

    # Save them to CSV
    # We'll flatten each confusion matrix as [TN, FP, FN, TP]
    with open(os.path.join(results_dir, f"confusion_matrix_{args.model}_{args.size}_train.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cm_train.flatten())
    with open(os.path.join(results_dir, f"confusion_matrix_{args.model}_{args.size}_val.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cm_val.flatten())
    with open(os.path.join(results_dir, f"confusion_matrix_{args.model}_{args.size}_test.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cm_test.flatten())

    # Save the fold-wise ROC/PR data so we can plot them with colored lines + mean
    fold_rocpr_file = os.path.join(results_dir, f"{args.model}_{args.size}_fold_rocpr.pkl")
    data_rocpr = {
        'fold_roc_test': fold_roc_test,
        'fold_pr_test': fold_pr_test
    }
    with open(fold_rocpr_file, 'wb') as pkl_file:
        pickle.dump(data_rocpr, pkl_file)

    # Summaries of fold-level results
    if len(fold_results) > 0:
        fold_results = np.array(fold_results)
        avg_train_loss      = np.mean(fold_results[:,0])
        avg_train_acc       = np.mean(fold_results[:,1])
        avg_train_prec      = np.mean(fold_results[:,2])
        avg_train_rec       = np.mean(fold_results[:,3])
        avg_train_f1        = np.mean(fold_results[:,4])

        avg_val_loss        = np.mean(fold_results[:,5])
        avg_val_acc         = np.mean(fold_results[:,6])
        avg_val_prec        = np.mean(fold_results[:,7])
        avg_val_rec         = np.mean(fold_results[:,8])
        avg_val_f1          = np.mean(fold_results[:,9])

        avg_test_loss       = np.mean(fold_results[:,10])
        avg_test_acc        = np.mean(fold_results[:,11])
        avg_test_prec       = np.mean(fold_results[:,12])
        avg_test_rec        = np.mean(fold_results[:,13])
        avg_test_f1         = np.mean(fold_results[:,14])

        print("Average over folds:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Train Prec: {avg_train_prec:.4f}, Train Rec: {avg_train_rec:.4f}, Train F1: {avg_train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, "
              f"Val Prec: {avg_val_prec:.4f}, Val Rec: {avg_val_rec:.4f}, Val F1: {avg_val_f1:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}, "
              f"Test Prec: {avg_test_prec:.4f}, Test Rec: {avg_test_rec:.4f}, Test F1: {avg_test_f1:.4f}")

        summary_file = os.path.join(results_dir, f"{args.model}_{args.size}_kfold_summary.csv")
        with open(summary_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Train Loss','Train Acc','Train Prec','Train Rec','Train F1',
                'Val Loss','Val Acc','Val Prec','Val Rec','Val F1',
                'Test Loss','Test Acc','Test Prec','Test Rec','Test F1'
            ])
            writer.writerow([
                avg_train_loss, avg_train_acc, avg_train_prec, avg_train_rec, avg_train_f1,
                avg_val_loss,   avg_val_acc,   avg_val_prec,   avg_val_rec,   avg_val_f1,
                avg_test_loss,  avg_test_acc,  avg_test_prec,  avg_test_rec,  avg_test_f1
            ])

    print("Done with all folds.")
