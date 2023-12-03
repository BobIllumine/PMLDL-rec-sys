from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_RATING_COL as RATING_COL,
    DEFAULT_PREDICTION_COL as PREDICT_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
    DEFAULT_TITLE_COL as TITLE_COL,
    SEED
)
import sklearn.preprocessing
from recommenders.datasets import movielens
from typing import Literal

def load_movielens(size: Literal['100k', '1m', '10m', '20m'], include_title=False):
    data = movielens.load_pandas_df(
        size=size,
        header=[USER_COL, ITEM_COL, RATING_COL],
        genres_col=ITEM_FEAT_COL,
        title_col=TITLE_COL if include_title else None
    )
    genres_encoder = sklearn.preprocessing.MultiLabelBinarizer()
    data[ITEM_FEAT_COL] = genres_encoder.fit_transform(
        data[ITEM_FEAT_COL].apply(lambda s: s.split("|"))
    ).tolist()
    return data