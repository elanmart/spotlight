import re
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder


#################
# USER FEATURES #
########################################################################################################################


def _indexer():
    indexer = defaultdict()
    indexer.default_factory = indexer.__len__

    return indexer


# We could use pandas, but pandas may cause python to segfault when used with pytorch
def _to_categorical(users_):
    IDs, *columns = zip(*users_)
    cat_columns = []

    for col in columns:
        indexer = _indexer()
        new_col = [indexer[item] for item in col]

        cat_columns += [new_col]

    new_users = list(zip(*cat_columns))

    return np.array(new_users)


def get_user_matrix(users_, include_zipcode=True):

    # add dummy user, because user indices start at 1
    users_.insert(0, users_[0])
    users_ = _to_categorical(users_)

    if not include_zipcode:
        users_ = users_[:, :-1]

    enc = OneHotEncoder()
    csr = enc.fit_transform(users_)

    return csr


##################
# MOVIE FEATURES #
########################################################################################################################

def _get_round(title, year, div):
    return f'round_{div}_', int(round(year / div))


def _get_int(title, year, div):
    return f'int_{div}_', int(year // div)


def _default_extractors():
    return [
        partial(_get_round, div=2),
        partial(_get_round, div=5),
        partial(_get_round, div=10),
        partial(_get_int, div=2),
        partial(_get_int, div=5),
        partial(_get_int, div=10),
    ]


def get_item_matrix(movies_, feature_extractors=None):

    feature_extractors = feature_extractors or _default_extractors()
    year_pattern       = re.compile("(.*)" + "\(" + "([0-9]{4})" + "\)")

    features = []

    for ID, title, genres, in movies_:

        line = {}

        match = re.match(year_pattern, title)
        (title,
         year) = match.groups()
        year = float(year)

        for genre in genres.split('|'):
            line[genre] = 1

        for extractor in feature_extractors:
            prefix, value = extractor(title, year)
            line[prefix + str(value)] = 1

        features.append(line)

    vectorizer = DictVectorizer()
    csr = vectorizer.fit_transform(features)

    return csr
