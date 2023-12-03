import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import scrapbook as sb
from tempfile import TemporaryDirectory
from recommenders.utils import tf_utils, gpu_utils, plot
import recommenders.evaluation.python_evaluation as evaluator
from recommenders.datasets.pandas_df_utils import user_item_pairs
import recommenders.models.wide_deep.wide_deep_utils as wide_deep
from recommenders.datasets.python_splitters import python_random_split
from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_RATING_COL as RATING_COL,
    DEFAULT_PREDICTION_COL as PREDICT_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
    SEED
)
from pathlib import Path
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

class WideDeepModel:
    #### Hyperparameters
    TOP_K = 10
    RANKING_METRICS = [
    evaluator.ndcg_at_k.__name__,
    evaluator.precision_at_k.__name__,
    ]
    RATING_METRICS = [
        evaluator.rmse.__name__,
        evaluator.mae.__name__,
    ]
    MODEL_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'wide_deep'
    MODEL_TYPE = "wide_deep"
    STEPS = 100  # Number of batches to train
    BATCH_SIZE = 32
    EVALUATE_WHILE_TRAINING = True
    # Wide (linear) model hyperparameters
    LINEAR_OPTIMIZER = "adagrad"
    LINEAR_OPTIMIZER_LR = 0.0621  # Learning rate
    LINEAR_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
    LINEAR_L2_REG = 0.0
    LINEAR_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
    # DNN model hyperparameters
    DNN_OPTIMIZER = "adadelta"
    DNN_OPTIMIZER_LR = 0.1
    DNN_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
    DNN_L2_REG = 0.0
    DNN_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
    # Layer dimensions. Defined as follows to make this notebook runnable from Hyperparameter tuning services like AzureML Hyperdrive
    DNN_HIDDEN_LAYER_1 = 0     # Set 0 to not use this layer
    DNN_HIDDEN_LAYER_2 = 64    # Set 0 to not use this layer
    DNN_HIDDEN_LAYER_3 = 128   # Set 0 to not use this layer
    DNN_HIDDEN_LAYER_4 = 512   # Note, at least one layer should have nodes.
    DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if h > 0]
    DNN_USER_DIM = 32          # User embedding feature dimension
    DNN_ITEM_DIM = 16          # Item embedding feature dimension
    DNN_DROPOUT = 0.8
    DNN_BATCH_NORM = 1         # 1 to use batch normalization, 0 if not.
    RANDOM_SEED = SEED

    def __init__(self, data, split_ratio=0.8) -> None:
        self.train, self.test = python_random_split(data, ratio=split_ratio, seed=self.RANDOM_SEED)
        # Data is needed on initialization to build feature columns
        if ITEM_FEAT_COL is None:
            self.items = data.drop_duplicates(ITEM_COL)[[ITEM_COL]].reset_index(drop=True)
            self.item_feat_shape = None
        else:
            self.items = data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True)
            self.item_feat_shape = len(self.items[ITEM_FEAT_COL][0])
        # Unique users in the dataset
        self.users = data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True)

        # Define wide (linear) and deep (dnn) features
        self.wide_columns, self.deep_columns = wide_deep.build_feature_columns(
            users=self.users[USER_COL].values,
            items=self.items[ITEM_COL].values,
            user_col=USER_COL,
            item_col=ITEM_COL,
            item_feat_col=ITEM_FEAT_COL,
            crossed_feat_dim=1000,
            user_dim=self.DNN_USER_DIM,
            item_dim=self.DNN_ITEM_DIM,
            item_feat_shape=self.item_feat_shape,
            model_type=self.MODEL_TYPE,
        )
        if self.MODEL_DIR is None:
            TMP_DIR = TemporaryDirectory()
            self.model_dir = TMP_DIR.name
        else:
            if os.path.exists(str(self.MODEL_DIR)) and os.listdir(str(self.MODEL_DIR)):
                raise ValueError(
                    "Model exists in {}. Use different directory name or "
                    "remove the existing checkpoint files first".format(str(self.MODEL_DIR))
                )
            TMP_DIR = None
            self.model_dir = str(self.MODEL_DIR)

        self.save_checkpoints_steps = max(1, self.STEPS // 5)
        self.model = wide_deep.build_model(
            model_dir=self.model_dir,
            wide_columns=self.wide_columns,
            deep_columns=self.deep_columns,
            linear_optimizer=tf_utils.build_optimizer(self.LINEAR_OPTIMIZER, self.LINEAR_OPTIMIZER_LR, **{
                'l1_regularization_strength': self.LINEAR_L1_REG,
                'l2_regularization_strength': self.LINEAR_L2_REG,
                'momentum': self.LINEAR_MOMENTUM,
            }),
            dnn_optimizer=tf_utils.build_optimizer(self.DNN_OPTIMIZER, self.DNN_OPTIMIZER_LR, **{
                'l1_regularization_strength': self.DNN_L1_REG,
                'l2_regularization_strength': self.DNN_L2_REG,
                'momentum': self.DNN_MOMENTUM,  
            }),
            dnn_hidden_units=self.DNN_HIDDEN_UNITS,
            dnn_dropout=self.DNN_DROPOUT,
            dnn_batch_norm=(self.DNN_BATCH_NORM==1),
            log_every_n_iter=max(1, self.STEPS//10),  # log 10 times
            save_checkpoints_steps=self.save_checkpoints_steps,
            seed=self.RANDOM_SEED
        )

        self.cols = {
            'col_user': USER_COL,
            'col_item': ITEM_COL,
            'col_rating': RATING_COL,
            'col_prediction': PREDICT_COL,
        }

        # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
        self.ranking_pool = user_item_pairs(
            user_df=self.users,
            item_df=self.items,
            user_col=USER_COL,
            item_col=ITEM_COL,
            user_item_filter_df=self.train,  # Remove seen items
            shuffle=True,
            seed=self.RANDOM_SEED
        )


    def train(self, num_steps=STEPS, batch_size=BATCH_SIZE, verbose=True):
        # Define training hooks to track performance while training
        hooks = []
        if self.EVALUATE_WHILE_TRAINING:
            evaluation_logger = tf_utils.MetricsLogger()
            for metrics in (self.RANKING_METRICS, self.RATING_METRICS):
                if len(metrics) > 0:
                    hooks.append(
                        tf_utils.evaluation_log_hook(
                            self.model,
                            logger=evaluation_logger,
                            true_df=self.test,
                            y_col=RATING_COL,
                            eval_df=self.ranking_pool if metrics==self.RANKING_METRICS else self.test.drop(RATING_COL, axis=1),
                            every_n_iter=self.save_checkpoints_steps,
                            model_dir=self.model_dir,
                            eval_fns=[evaluator.metrics[m] for m in metrics],
                            **({**self.cols, 'k': self.TOP_K} if metrics==self.RANKING_METRICS else self.cols)
                        )
                    )

        # Define training input (sample feeding) function
        self.train_fn = tf_utils.pandas_input_fn(
            df=self.train,
            y_col=RATING_COL,
            batch_size=self.BATCH_SIZE,
            num_epochs=None,  # We use steps=TRAIN_STEPS instead.
            shuffle=True,
            seed=self.RANDOM_SEED,
        )
        print(
            "Training steps = {}, Batch size = {} (num epochs = {})"
            .format(self.STEPS, self.BATCH_SIZE, (self.STEPS*self.BATCH_SIZE)//len(self.train))
        )

        try:
            self.model.train(
                input_fn=self.train_fn,
                hooks=hooks,
                steps=self.STEPS
            )
        except tf.train.NanLossDuringTrainingError:
            import warnings
            warnings.warn(
                "Training stopped with NanLossDuringTrainingError. "
                "Try other optimizers, smaller batch size and/or smaller learning rate."
            )
        if self.EVALUATE_WHILE_TRAINING:
            logs = evaluation_logger.get_log()
            for i, (m, v) in enumerate(logs.items(), 1):
                sb.glue("eval_{}".format(m), v)
                x = [self.save_checkpoints_steps*i for i in range(1, len(v)+1)]
                plot.line_graph(
                    values=list(zip(v, x)),
                    labels=m,
                    x_name="steps",
                    y_name=m,
                    subplot=(math.ceil(len(logs)/2), 2, i),
                )
            plot.savefig(str(Path(__file__).parent.parent.parent.parent / 'reports' / 'figures' / 'training_loss.png'))

    def top_k(self, k=TOP_K):
        if len(self.RANKING_METRICS) > 0:
            predictions = list(self.model.predict(input_fn=tf_utils.pandas_input_fn(df=self.ranking_pool)))
            prediction_df = self.ranking_pool.copy()
            prediction_df[PREDICT_COL] = [p['predictions'][0] for p in predictions]

            ranking_results = {}
            for m in self.RANKING_METRICS:
                result = evaluator.metrics[m](self.test, prediction_df, **{**self.cols, 'k': k})
                sb.glue(m, result)
                ranking_results[m] = result
            return prediction_df, ranking_results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        exported_path = tf_utils.export_model(
            model=self.model,
            train_input_fn=self.train_fn,
            eval_input_fn=tf_utils.pandas_input_fn(
                df=self.test, y_col=RATING_COL
            ),
            tf_feat_cols=self.wide_columns + self.deep_columns,
            base_dir=path
        )
        sb.glue('saved_model_dir', str(exported_path))   