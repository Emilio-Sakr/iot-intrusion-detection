"""
Composite classifiers used at serving time.

The hierarchical pipelines in this module are pickled by joblib and reloaded by
the FastAPI service. Classes live here (not inside a notebook) so joblib stores
their module path as `src.pipelines.<ClassName>` and `joblib.load` can resolve
them from any process that has the project on its Python path.
"""
from __future__ import annotations

import numpy as np


class ThreeStageClassifier:
    """
    Hierarchical intrusion detection pipeline.

        Stage 1 — binary BenignTraffic vs. Attack
        Stage 2 — attack family (e.g. DDoS, DoS, Mirai, Recon, Spoofing, ...)
        Stage 3 — per-family fitted classifier that picks the specific attack

    Exposes the sklearn-like ``predict`` and ``predict_proba`` surface so it
    drops into the same evaluation and serving code as any flat classifier.
    """

    BENIGN_LABEL = "BenignTraffic"
    ATTACK_LABEL = "Attack"

    def __init__(self, stage1, stage2, stage3, threshold: float = 0.3):
        self.stage1 = stage1
        self.stage2 = stage2
        # dict[family_name -> fitted estimator | str constant label]
        self.stage3 = stage3
        self.threshold = threshold

    # -- class metadata -----------------------------------------------------

    # Expose as a property (not an __init__ assignment) so that instances
    # pickled before this attribute existed still work after unpickle —
    # __init__ is not re-run on load, but the class's descriptors are.
    @property
    def classes_(self) -> np.ndarray:
        return self._build_class_order()

    def _build_class_order(self) -> np.ndarray:
        attack_labels: list[str] = []
        for estimator in self.stage3.values():
            if isinstance(estimator, str):
                attack_labels.append(estimator)
            else:
                attack_labels.extend(str(c) for c in estimator.classes_)
        seen: set[str] = set()
        unique: list[str] = []
        for label in attack_labels:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        return np.array([self.BENIGN_LABEL] + unique)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _slice(X, mask):
        return X.iloc[mask] if hasattr(X, "iloc") else X[mask]

    # -- sklearn-like interface --------------------------------------------

    def predict(self, X):
        proba = self.stage1.predict_proba(X)
        attack_idx = list(self.stage1.classes_).index(self.ATTACK_LABEL)
        attack_mask = proba[:, attack_idx] >= self.threshold

        y_pred = np.empty(len(attack_mask), dtype=object)
        y_pred[:] = self.BENIGN_LABEL

        if not attack_mask.any():
            return y_pred

        X_attack = self._slice(X, attack_mask)
        family_pred = self.stage2.predict(X_attack)

        attack_pred = np.empty(len(family_pred), dtype=object)
        for family, estimator in self.stage3.items():
            fam_mask = family_pred == family
            if not fam_mask.any():
                continue
            if isinstance(estimator, str):
                attack_pred[fam_mask] = estimator
            else:
                X_fam = self._slice(X_attack, fam_mask)
                attack_pred[fam_mask] = estimator.predict(X_fam)

        y_pred[attack_mask] = attack_pred
        return y_pred

    def predict_proba(self, X):
        """
        Marginal class probabilities:

            P(class | x) = P(benign | x)                                        for benign
                         = P(attack | x) * P(family | x) * P(class | family, x) otherwise

        Stage 2 and Stage 3 probas are computed on all rows (not just rows
        Stage 1 flagged as attack); we weight by ``P(attack | x)`` so
        benign-dominated rows contribute ~zero to attack-class probabilities.
        Simpler code, the cost is one extra matrix multiply on benign rows.
        """
        n = len(X)
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        out = np.zeros((n, len(self.classes_)), dtype=np.float64)

        # Stage 1
        s1_proba = self.stage1.predict_proba(X)
        s1_classes = list(self.stage1.classes_)
        benign_idx_s1 = s1_classes.index(self.BENIGN_LABEL)
        attack_idx_s1 = s1_classes.index(self.ATTACK_LABEL)
        p_benign = s1_proba[:, benign_idx_s1]
        p_attack = s1_proba[:, attack_idx_s1]

        out[:, class_to_idx[self.BENIGN_LABEL]] = p_benign

        # Stage 2 (conditional on attack)
        s2_proba = self.stage2.predict_proba(X)
        s2_families = list(self.stage2.classes_)

        # Stage 3 per family
        for family_idx, family in enumerate(s2_families):
            estimator = self.stage3.get(family)
            if estimator is None:
                continue
            p_family = p_attack * s2_proba[:, family_idx]

            if isinstance(estimator, str):
                out[:, class_to_idx[estimator]] += p_family
                continue

            s3_proba = estimator.predict_proba(X)
            for local_idx, cls in enumerate(estimator.classes_):
                cls_str = str(cls)
                if cls_str in class_to_idx:
                    out[:, class_to_idx[cls_str]] += p_family * s3_proba[:, local_idx]

        return out
