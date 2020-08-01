from typing import Union

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf(TfidfVectorizer):
    __doc__ = super.__doc__

    def fit(self, raw_documents, y=None):
        raw_documents = self._prepare_documents(raw_documents)
        return super(TfIdf, self).fit(raw_documents, y)

    def fit_transform(self, raw_documents, y=None):
        raw_documents = self._prepare_documents(raw_documents)
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def transform(self, raw_documents, copy="deprecated"):
        raw_documents = self._prepare_documents(raw_documents)
        return super(TfIdf, self).transform(raw_documents, copy)

    @staticmethod
    def _prepare_documents(raw_documents: Union[pd.Series, pd.DataFrame]):
        if isinstance(raw_documents, pd.DataFrame):
            raw_documents = raw_documents.sum(axis=1)
        return raw_documents
