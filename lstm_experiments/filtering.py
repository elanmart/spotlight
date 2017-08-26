import numpy as np


def split_ratings(ratings_):
    (user_ids,
     item_ids,
     ratings,
     timestamps) = (ratings_[:, i]
                    for i in range(4))

    return user_ids, item_ids, ratings, timestamps


def filer_negative_interactions(ratings_,
                                min_rating=4, min_interactions=20, split=True):

    stars   = ratings_[:, 2]

    mask    = (stars >= min_rating)
    ratings_ = ratings_[mask, :]

    user_ids = ratings_[:, 0]

    (__,
     inverse,
     n_interactions) = np.unique(user_ids,
                                 return_inverse=True, return_counts=True)

    enough_counts = (n_interactions >= min_interactions)
    mask          = enough_counts[inverse]
    ratings_       = ratings_[mask, :]

    if split:
        return split_ratings(ratings_)

    return ratings_
