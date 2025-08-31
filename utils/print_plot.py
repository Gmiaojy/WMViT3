import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Redirect stdout and stderr to the logger
class StreamToLogger(object):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
    def flush(self):
        pass
    
def setup_logging(output_dir, logger_name='training_log.txt'):
    """Configures logging to print to console and save to a file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, logger_name)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    # Create logger instance and redirect stdout and stderr
    stdout_logger = logging.getLogger('STDOUT')
    sl_stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl_stdout
    stderr_logger = logging.getLogger('STDERR')
    sl_stderr = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl_stderr
    
def log_training_config(config_dict):
    """
    Record a structured configuration dictionary
    """
    logging.info("\n" + "=" * 10 + " Training Configuration Overview " + "=" * 10)
    for section, settings in config_dict.items():
        logging.info(f"--- {section} ---")
        max_key_len = max(len(key) for key in settings.keys()) if settings else 0
        for key, value in settings.items():
            
            logging.info(f"  {key:<{max_key_len}} : {value}")
    logging.info("=" * 81 + "\n")


def plot_confusion_matrix(cm, class_names, save_path, figsize=(12, 10), normalize=False,
                          tick_fontsize=14, label_fontsize=14,
                          annot_fontsize=16, title_fontsize=16,
                          cbar_tick_fontsize=14, 
                          label_pad=20,           
                          title_pad=20,
                          cmap='YlGnBu',           
                          title=None):
    """
    Draw and save the confusion matrix heatmap
    """
    try:
        plt.figure(figsize=figsize)
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            data_to_plot, fmt, default_title = cm_normalized, '.2f', 'Normalized Confusion Matrix'
        else:
            data_to_plot, fmt, default_title = cm, 'd', 'Confusion Matrix'

        current_title = title if title is not None else default_title
        
        ax = sns.heatmap(data_to_plot, annot=True, xticklabels=class_names, yticklabels=class_names,
                         cmap=cmap, cbar=True, fmt=fmt, annot_kws={"size": annot_fontsize})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        plt.xlabel('Predicted Label', fontsize=label_fontsize, labelpad=label_pad)
        plt.ylabel('True Label', fontsize=label_fontsize, labelpad=label_pad)
        plt.title(current_title, fontsize=title_fontsize, pad=title_pad)

        plt.xticks(rotation=0, ha='center', fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='jpg')
        logging.info(f"\nConfusion matrix saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to plot or save confusion matrix: {e}", exc_info=True)
    finally:
        plt.close()


def plot_metrics_trend(metrics_history, save_path, figsize=(12, 8),
                       tick_fontsize=12, label_fontsize=14,
                       legend_fontsize=12, title_fontsize=16,
                       title=None):
    """
    Plot the change curves of precision, recall and F1 score
    """
    try:
        if not all(key in metrics_history for key in ['precision', 'recall', 'f1']) or not metrics_history['f1']:
            logging.warning("Metrics history is missing data. Skipping plotting P/R/F1 trend.")
            return

        plt.figure(figsize=figsize)
        epochs = range(1, len(metrics_history['f1']) + 1)

        plt.plot(epochs, metrics_history['precision'], 'g-', label='Precision (macro)')
        plt.plot(epochs, metrics_history['recall'], 'r-.', label='Recall (macro)')
        plt.plot(epochs, metrics_history['f1'], 'm:', label='F1 Score (macro)')

        current_title = title if title is not None else 'Training Metrics Trend (P/R/F1 Macro)'
        plt.title(current_title, fontsize=title_fontsize)
        plt.xlabel('Epochs', fontsize=label_fontsize)
        plt.ylabel('Score', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.ylim(0, 1.05)
        plt.legend(loc='best', fontsize=legend_fontsize)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='jpg')
        logging.info(f"Training metrics trend curves saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to plot or save metrics trend: {e}", exc_info=True)
    finally:
        plt.close()


def plot_training_curves(history, save_path, figsize=(12, 8),
                         tick_fontsize=12, label_fontsize=14,
                         legend_fontsize=12, title_fontsize=16,
                         title=None):
    """
    Plot the loss and accuracy curves during the training process, supporting dual Y - axes.
    """
    try:
        if not history or not history.get('train_loss'):
            logging.warning("History is empty or missing 'train_loss'. Skipping plotting training curves.")
            return

        num_epochs_run = len(history['train_loss'])
        epochs = range(1, num_epochs_run + 1)

        plt.figure(figsize=figsize)
        ax1 = plt.gca()
        line1, = ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epochs', fontsize=label_fontsize)
        ax1.set_ylabel('Loss', color='b', fontsize=label_fontsize)
        ax1.tick_params(axis='y', labelcolor='b', labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        line2, = ax2.plot(epochs, history['train_acc'], 'r-.', label='Train Acc')
        lines_for_legend = [line1, line2]

        if 'val_acc' in history and any(acc is not None for acc in history['val_acc']):
            line3, = ax2.plot(epochs, history['val_acc'], 'm:', label='Valid Acc')
            lines_for_legend.append(line3)

        ax2.set_ylabel('Accuracy', color='r', fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=tick_fontsize)
        ax2.set_ylim(0, 1.05)

        labels_for_legend = [l.get_label() for l in lines_for_legend]
        ax1.legend(lines_for_legend, labels_for_legend, loc='best', fontsize=legend_fontsize)

        current_title = title if title is not None else 'Training Loss and Accuracy Curves'
        plt.title(current_title, fontsize=title_fontsize)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='jpg')
        logging.info(f"Training curves saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to plot or save training curves: {e}", exc_info=True)
    finally:
        plt.close()