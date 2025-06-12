"""
General Fine-Tuning Pipeline for Helical Platform
Supports both Classification and Regression tasks with Geneformer
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import warnings
import logging

# Helical imports
from helical.utils import get_anndata_from_hf_dataset
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from datasets import load_dataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


class HelicalFineTuningPipeline:
    """
    A comprehensive pipeline for fine-tuning Geneformer models on classification and regression tasks.
    
    Supports:
    - Classification tasks (cell type prediction, disease classification, etc.)
    - Regression tasks (gene expression prediction, continuous phenotype prediction, etc.)
    """
    
    def __init__(self, 
                 task_type: str = "classification",
                 model_name: str = "gf-12L-95M-i4096",
                 batch_size: int = 10,
                 device: str = None):
        """
        Initialize the pipeline.
        
        Args:
            task_type: Either "classification" or "regression"
            model_name: Geneformer model variant to use
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.task_type = task_type.lower()
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialize placeholders
        self.model = None
        self.class_id_dict = None
        self.id_class_dict = None
        self.train_processed = None
        self.test_processed = None
        
        print(f"Pipeline initialized for {task_type} task using {model_name} on {self.device}")
    
    def load_data(self, 
                  train_data, 
                  test_data=None, 
                  label_column: str = None,
                  dataset_name: str = None) -> Tuple:
        """
        Load and prepare data from various sources.
        
        Args:
            train_data: Training data (AnnData, HF dataset name, or file path)
            test_data: Test data (optional, same formats as train_data)
            label_column: Column name containing labels/targets
            dataset_name: If loading from Hugging Face, specify dataset name
        
        Returns:
            Tuple of (train_dataset, test_dataset, labels_info)
        """
        # Handle different data input types
        if isinstance(train_data, str) and dataset_name:
            # Load from Hugging Face
            ds = load_dataset(train_data, trust_remote_code=True, download_mode="reuse_cache_if_exists")
            train_dataset = get_anndata_from_hf_dataset(ds["train"])
            test_dataset = get_anndata_from_hf_dataset(ds["test"]) if "test" in ds else None
        else:
            # Assume AnnData objects
            train_dataset = train_data
            test_dataset = test_data
        
        if label_column is None:
            raise ValueError("label_column must be specified")
        
        # Extract labels
        train_labels = list(train_dataset.obs[label_column])
        test_labels = list(test_dataset.obs[label_column]) if test_dataset is not None else []
        
        # Process labels based on task type
        if self.task_type == "classification":
            train_labels, test_labels, labels_info = self._process_classification_labels(
                train_labels, test_labels
            )
        else:  # regression
            train_labels, test_labels, labels_info = self._process_regression_labels(
                train_labels, test_labels
            )
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.labels_info = labels_info
        
        print(f"Data loaded successfully:")
        print(f"  Train samples: {len(train_dataset)}")
        if test_dataset is not None:
            print(f"  Test samples: {len(test_dataset)}")
        print(f"  Task: {self.task_type}")
        if self.task_type == "classification":
            print(f"  Number of classes: {labels_info['num_classes']}")
        
        return train_dataset, test_dataset, labels_info
    
    def _process_classification_labels(self, train_labels: List, test_labels: List) -> Tuple:
        """Process labels for classification tasks."""
        # Create label mapping
        all_labels = set(train_labels) | set(test_labels)
        self.class_id_dict = dict(zip(all_labels, range(len(all_labels))))
        self.id_class_dict = {v: k for k, v in self.class_id_dict.items()}
        
        # Convert to integer labels
        train_labels_int = [self.class_id_dict[label] for label in train_labels]
        test_labels_int = [self.class_id_dict[label] for label in test_labels]
        
        labels_info = {
            'num_classes': len(all_labels),
            'class_names': list(all_labels),
            'class_mapping': self.class_id_dict
        }
        
        return train_labels_int, test_labels_int, labels_info
    
    def _process_regression_labels(self, train_labels: List, test_labels: List) -> Tuple:
        """Process labels for regression tasks."""
        # Convert to float and handle missing values
        train_labels_float = [float(label) if label is not None else 0.0 for label in train_labels]
        test_labels_float = [float(label) if label is not None else 0.0 for label in test_labels]
        
        labels_info = {
            'min_value': min(train_labels_float + test_labels_float),
            'max_value': max(train_labels_float + test_labels_float),
            'mean_value': np.mean(train_labels_float)
        }
        
        return train_labels_float, test_labels_float, labels_info
    
    def setup_model(self, **config_kwargs) -> None:
        """
        Setup the Geneformer fine-tuning model.
        
        Args:
            **config_kwargs: Additional configuration parameters for GeneformerConfig
        """
        # Default configuration
        default_config = {
            'device': self.device,
            'batch_size': self.batch_size,
            'model_name': self.model_name
        }
        default_config.update(config_kwargs)
        
        # Create model configuration
        geneformer_config = GeneformerConfig(**default_config)
        
        # Determine output size
        if self.task_type == "classification":
            output_size = self.labels_info['num_classes']
        else:  # regression
            output_size = 1  # Single continuous output
        
        # Create fine-tuning model
        self.model = GeneformerFineTuningModel(
            geneformer_config=geneformer_config,
            fine_tuning_head=self.task_type,
            output_size=output_size
        )
        
        print(f"Model setup complete. Output size: {output_size}")
    
    def process_data(self, gene_names: str = "index") -> None:
        """
        Process the data for Geneformer training.
        
        Args:
            gene_names: Gene naming convention in the data
        """
        if self.model is None:
            raise ValueError("Model must be setup before processing data. Call setup_model() first.")
        
        # Process training data
        self.train_processed = self.model.process_data(self.train_dataset, gene_names=gene_names)
        self.train_processed = self.train_processed.add_column("labels", self.train_labels)
        
        # Process test data if available
        if self.test_dataset is not None:
            self.test_processed = self.model.process_data(self.test_dataset, gene_names=gene_names)
            self.test_processed = self.test_processed.add_column("labels", self.test_labels)
        
        print("Data processing complete.")
    
    def train(self, 
              epochs: int = 5,
              learning_rate: float = 1e-4,
              freeze_layers: int = 2,
              validation_split: float = 0.2,
              **train_kwargs) -> None:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            freeze_layers: Number of layers to freeze during training
            validation_split: Fraction of training data to use for validation
            **train_kwargs: Additional training parameters
        """
        if self.train_processed is None:
            raise ValueError("Data must be processed before training. Call process_data() first.")
        
        # Setup loss function based on task type
        if self.task_type == "classification":
            loss_function = torch.nn.CrossEntropyLoss()
        else:  # regression
            loss_function = torch.nn.MSELoss()
        
        # Default training parameters
        default_params = {
            'train_dataset': self.train_processed.shuffle(),
            'validation_dataset': self.test_processed,
            'label': "labels",
            'epochs': epochs,
            'freeze_layers': freeze_layers,
            'optimizer_params': {"lr": learning_rate},
            'loss_function': loss_function,
            'lr_scheduler_params': {
                "name": "linear", 
                "num_warmup_steps": 0, 
                'num_training_steps': epochs
            }
        }
        default_params.update(train_kwargs)
        
        # Train the model
        print(f"Starting training for {epochs} epochs...")
        self.model.train(**default_params)
        print("Training complete!")
    
    def evaluate(self) -> Dict:
        """
        Evaluate the trained model and return metrics.
        
        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        if self.test_processed is None:
            raise ValueError("Test data not available for evaluation.")
        
        # Get predictions
        outputs = self.model.get_outputs(self.test_processed)
        embeddings = self.model.get_embeddings(self.test_processed)
        
        results = {
            'outputs': outputs,
            'embeddings': embeddings,
            'true_labels': self.test_labels
        }
        
        if self.task_type == "classification":
            predictions = outputs.argmax(axis=1)
            results.update(self._evaluate_classification(predictions))
        else:  # regression
            predictions = outputs.flatten()
            results.update(self._evaluate_regression(predictions))
        
        results['predictions'] = predictions
        return results
    
    def _evaluate_classification(self, predictions: np.ndarray) -> Dict:
        """Evaluate classification results."""
        # Calculate metrics
        report = classification_report(
            self.test_labels, predictions, 
            target_names=[self.id_class_dict[i] for i in range(len(self.id_class_dict))],
            output_dict=True
        )
        
        cm = confusion_matrix(self.test_labels, predictions)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy']
        }
    
    def _evaluate_regression(self, predictions: np.ndarray) -> Dict:
        """Evaluate regression results."""
        mse = mean_squared_error(self.test_labels, predictions)
        r2 = r2_score(self.test_labels, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'mae': np.mean(np.abs(np.array(self.test_labels) - predictions))
        }
    
    def visualize_results(self, results: Dict, save_path: str = None) -> None:
        """
        Create visualizations for the results.
        
        Args:
            results: Results dictionary from evaluate()
            save_path: Optional path to save the plots
        """
        if self.task_type == "classification":
            self._plot_classification_results(results, save_path)
        else:
            self._plot_regression_results(results, save_path)
    
    def _plot_classification_results(self, results: Dict, save_path: str = None) -> None:
        """Create classification visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. UMAP of embeddings
        reducer = umap.UMAP(min_dist=0.2, n_components=2, n_neighbors=4)
        embedding_2d = reducer.fit_transform(results['embeddings'])
        
        plot_df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
        plot_df['True_Labels'] = [self.id_class_dict[label] for label in results['true_labels']]
        plot_df['Predicted_Labels'] = [self.id_class_dict[label] for label in results['predictions']]
        
        sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='True_Labels', 
                       ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('UMAP - True Labels')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='Predicted_Labels', 
                       ax=axes[0,1], alpha=0.7)
        axes[0,1].set_title('UMAP - Predicted Labels')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Confusion Matrix
        cm_normalized = results['confusion_matrix'].astype('float') / \
                       results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
        
        class_names = [self.id_class_dict[i] for i in range(len(self.id_class_dict))]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1,0])
        axes[1,0].set_title('Normalized Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('True')
        
        # 3. Classification Report Visualization
        report_df = pd.DataFrame(results['classification_report']).iloc[:-1, :].T
        report_df = report_df.drop(['support'], axis=1)
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1,1])
        axes[1,1].set_title('Classification Metrics')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_regression_results(self, results: Dict, save_path: str = None) -> None:
        """Create regression visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        true_values = np.array(results['true_labels'])
        predictions = results['predictions']
        
        # 1. Scatter plot of predictions vs true values
        axes[0,0].scatter(true_values, predictions, alpha=0.6)
        axes[0,0].plot([true_values.min(), true_values.max()], 
                      [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('True Values')
        axes[0,0].set_ylabel('Predictions')
        axes[0,0].set_title(f'Predictions vs True Values (R² = {results["r2_score"]:.3f})')
        
        # 2. Residuals plot
        residuals = true_values - predictions
        axes[0,1].scatter(predictions, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predictions')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals Plot')
        
        # 3. Distribution of residuals
        axes[1,0].hist(residuals, bins=30, alpha=0.7)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Residuals')
        
        # 4. UMAP of embeddings
        reducer = umap.UMAP(min_dist=0.2, n_components=2, n_neighbors=4)
        embedding_2d = reducer.fit_transform(results['embeddings'])
        
        scatter = axes[1,1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=true_values, cmap='viridis', alpha=0.6)
        axes[1,1].set_xlabel('UMAP1')
        axes[1,1].set_ylabel('UMAP2')
        axes[1,1].set_title('UMAP - Colored by True Values')
        plt.colorbar(scatter, ax=axes[1,1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def run_complete_pipeline(self, 
                            train_data,
                            test_data=None,
                            label_column: str = None,
                            dataset_name: str = None,
                            epochs: int = 5,
                            learning_rate: float = 1e-4,
                            **kwargs) -> Dict:
        """
        Run the complete pipeline from data loading to evaluation.
        
        Args:
            train_data: Training data
            test_data: Test data (optional)
            label_column: Column containing labels/targets
            dataset_name: HF dataset name if applicable
            epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional parameters
        
        Returns:
            Complete results dictionary
        """
        print("🚀 Starting Helical Fine-Tuning Pipeline")
        print("=" * 50)
        
        # 1. Load data
        self.load_data(train_data, test_data, label_column, dataset_name)
        
        # 2. Setup model
        self.setup_model(**kwargs.get('model_config', {}))
        
        # 3. Process data
        self.process_data(**kwargs.get('process_config', {}))
        
        # 4. Train
        self.train(epochs=epochs, learning_rate=learning_rate, 
                  **kwargs.get('train_config', {}))
        
        # 5. Evaluate
        results = self.evaluate()
        
        # 6. Visualize
        self.visualize_results(results, kwargs.get('save_path'))
        
        print("✅ Pipeline completed successfully!")
        return results


# Example usage functions
def run_classification_example():
    """Example of running classification pipeline."""
    pipeline = HelicalFineTuningPipeline(
        task_type="classification",
        model_name="gf-6L-30M-i2048",
        batch_size=10
    )
    
    results = pipeline.run_complete_pipeline(
        train_data="helical-ai/yolksac_human",
        dataset_name="helical-ai/yolksac_human",
        label_column="LVL1",
        epochs=3,
        learning_rate=1e-4
    )
    
    return results

def run_regression_example():
    """Example of running regression pipeline."""
    pipeline = HelicalFineTuningPipeline(
        task_type="regression",
        model_name="gf-12L-95M-i4096",
        batch_size=8
    )
    
    # For regression, you would load your own data with continuous targets
    # results = pipeline.run_complete_pipeline(
    #     train_data=your_train_data,
    #     test_data=your_test_data,
    #     label_column="continuous_target",
    #     epochs=5,
    #     learning_rate=5e-5
    # )
    
    return pipeline

if __name__ == "__main__":
    # Run classification example
    print("Running classification example...")
    classification_results = run_classification_example()
    
    # Print summary
    if 'accuracy' in classification_results:
        print(f"\nClassification Accuracy: {classification_results['accuracy']:.3f}")