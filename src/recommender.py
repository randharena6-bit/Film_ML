from pathlib import Path
from typing import List, Dict, Optional, Required

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(
        self,
        data_directory: str = "data/ml-latest-small"
    ):
        self.data_directory = Path(data_directory)

        self.movie_path = self.data_directory / "movie.csv"
        self.ratings_path = self.data_directory / "ratings.csv"
        self.tags_path = self.data_directory / "tags.csv"

        self._check_files()
        self._load_data()
        self._prepare_features()
        self._build_similarity_matrix()

    def _check_files(self) -> None:
        required_files = [
            self.movie_path,
            self.ratings_path,
            self.tags_path
        ]

        missing_files = [
            str(file)
            for file in required_files
            if not file.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Fichier manquants : " + ",".join(missing_files)
            )
