import numpy as np

from certaudit.belief import Structure, train_qmodels
from certaudit.generator import make_split
from certaudit.oracle import make_view


def test_fast_mlp_matches_sklearn():
    qm = train_qmodels(make_split(0, 1), n_masks=2)
    st = Structure(make_view(make_split(1, 1)[0]))
    C, tp = st.context({})
    for role, m in qm.models.items():
        X = st.features(role, C, tp)
        if len(X):
            assert np.allclose(qm.predict(role, X), m.predict_proba(X)[:, 1], atol=1e-9)
