# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Author disambiguation -- Clustering.

See README.rst for further details.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import argparse
import pickle
import json
import numpy as np

from sklearn.cross_validation import train_test_split

# These imports are used during unpickling.
from utils import get_author_full_name
from utils import get_author_other_names
from utils import get_author_initials
from utils import get_author_affiliation
from utils import get_title
from utils import get_journal
from utils import get_abstract
from utils import get_coauthors_from_range
from utils import get_keywords
from utils import get_collaborations
from utils import get_references
from utils import get_year
from utils import group_by_signature
from utils import load_signatures

from beard.clustering import BlockClustering
from beard.clustering import block_last_name_first_initial


def _affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]

    return distances


def blocking(input_signatures, input_records, distance_model,
             blocks_directory):
    """Create blocks and dump themto files.

    Parameters
    ----------
    :param input_signatures: string
        Path to the file with signatures. The content should be a JSON array
        of dictionaries holding metadata about signatures.

        [{"signature_id": 0,
          "author_name": "Doe, John",
          "publication_id": 10, ...}, { ... }, ...]

    :param input_records: string
        Path to the file with records. The content should be a JSON array of
        dictionaries holding metadata about records

        [{"publication_id": 0,
          "title": "Author disambiguation using Beard", ... }, { ... }, ...]

    :param distance_model: string
        Path to the file with the distance model. The file should be a pickle
        created using the ``distance.py`` script.

    """
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    distance_estimator = pickle.load(open(distance_model, "rb"))
    signatures, records = load_signatures(input_signatures,
                                          input_records)

    indices = {}
    X = np.empty((len(signatures), 1), dtype=np.object)
    for i, signature in enumerate(sorted(signatures.values(),
                                         key=lambda s: s["signature_id"])):
        X[i, 0] = signature
        indices[signature["signature_id"]] = i

    # Semi-supervised block clustering
    if input_clusters:
        true_clusters = json.load(open(input_clusters, "r"))
        y_true = -np.ones(len(X), dtype=np.int)

        for label, signature_ids in true_clusters.items():
            for signature_id in signature_ids:
                y_true[indices[signature_id]] = label

        if clustering_test_size is not None:
            train, test = train_test_split(
                np.arange(len(X)),
                test_size=clustering_test_size,
                random_state=clustering_random_state)

            y = -np.ones(len(X), dtype=np.int)
            y[train] = y_true[train]

        else:
            y = y_true

    else:
        y = None

    Blocking(blocking=block_last_name_first_initial,
             blocks_directory=blocks_directory).block(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_model", required=True, type=str)
    parser.add_argument("--input_signatures", required=True, type=str)
    parser.add_argument("--input_records", required=True, type=str)
    parser.add_argument("--blocks_directory", required=True, type=str)
    args = parser.parse_args()

    blocking(args.input_signatures, args.input_records, args.distance_model,
             args.blocks_directory)
