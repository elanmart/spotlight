import os
import zipfile

import numpy as np

from spotlight.datasets._transport import download


DATASET = 'ml-1m'
FILE    = DATASET + '.zip'
URL     = "http://files.grouplens.org/datasets/movielens/" + FILE

USERS   = DATASET + '/users.dat'
MOVIES  = DATASET + '/movies.dat'
RATINGS = DATASET + '/ratings.dat'


def _perhaps_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def _read(archive, fname):

    lines = archive.read(fname).decode(errors='ignore').split('\n')

    items = [line.split('::')
             for line in lines
             if len(line) > 0]

    items = [[_perhaps_int(x)
              for x in item]
             for item in items]

    return items


def _read_movielens(path):

    with zipfile.ZipFile(path) as movielens:
        users_ = _read(movielens, USERS)
        movies_ = _read(movielens, MOVIES)
        ratings_ = _read(movielens, RATINGS)

        # columns: [user_id, item_id, stars (1-5), timestamp]
        ratings_ = np.array(ratings_)

        return users_, movies_, ratings_


def read_or_get_movielens(directory):

    path = os.path.join(directory, FILE)

    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        download(URL, path)

    return _read_movielens(path=path)
