"""FNR: factor neighborhoods, window statistics and constrained ranking."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import ThreadpoolController

from ..features.bounded import build_features
from ..features.window import panels
from ..features.window_math import stats, solve_at
from ..features.window_state import state_panel
from ..models import peer
from . import base as base_module
from .rank_merge import merge_scores
from .window_model import FrozenComponents, rank


class InferenceModel(FrozenComponents):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        policy = json.loads((Path(model_dir) / "ranking_policy.json").read_text())
        if policy != {
            "head_count": 60,
            "head_auxiliary_weights": [0.5, 0.5],
            "tail_signal": "K3_R80",
        }:
            raise ValueError("Unexpected ranking policy")
        self.head_count = policy["head_count"]
        self._threadpools = ThreadpoolController()

    def predict(self, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Score the as-of universe from the current 120-day input window."""
        with self._threadpools.limit(limits=1):
            stamp = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
            result = self._predict(stamp, data)
            protected = result[["date", "code", "rh50_score"]].rename(
                columns={"rh50_score": "score"}
            )
            return merge_scores(
                protected, result[["date", "code", "score"]], self.head_count
            )

    def _predict(self, stamp, data):
        # Restrict every component to the same visible exchange dates.
        day = int(stamp.replace("-", ""))
        dates = np.sort(
            data["MarketData"].loc[lambda x: x["日期"].le(day), "日期"].unique()
        )[-120:]
        data = {key: frame.loc[frame["日期"].isin(dates)].copy()
                for key, frame in data.items()}
        data["Factor33_winsor"]["is_st"] = 0.0
        if len(dates) != 120 or dates[-1] != day:
            raise ValueError("Expected 120 visible exchange dates")

        # The frozen ensemble and the two neighborhood heads share today's encoding.
        base, cur = base_module.InferenceModel.predict(
            self.base, stamp, data, return_features=True
        )
        x0 = self.standardize(cur)
        mask0 = np.ones((1, len(cur)), bool)
        n0, teacher0, graph0 = peer.encode(self.teacher, x0[None], mask0)
        idx, weights = graph0["factor_barra"]
        f0 = peer.features(n0, idx, weights, "factor_barra")[None]
        a = peer.apply_adapter(self.p01, f0, mask0, teacher0[None])[0]
        b = peer.apply_adapter(self.r1, f0, mask0, teacher0[None])[0]
        aux0 = (0.475 * a.astype(float) + 0.525 * b.astype(float)).astype(np.float32)
        head_aux = (0.5 * a.astype(float) + 0.5 * b.astype(float)).astype(np.float32)
        rh50_score = np.round(
            0.6 * rank(base.set_index("code").loc[cur["代码"]].score.to_numpy())
            + 0.4 * rank(head_aux[0]), 12,
        )
        raw_base = np.round(
            0.6 * rank(base.set_index("code").loc[cur["代码"]].score.to_numpy())
            + 0.4 * rank(aux0[0]), 12,
        )

        # Raw-factor statistics use only labels that have matured within the window.
        p = panels(data, stamp)
        t = len(p["days"]) - 1
        keep = self.kcfg["keep"]
        xx = p["x"][:, :, keep]
        curix = np.searchsorted(p["codes"], cur["代码"].to_numpy())
        assert np.array_equal(p["codes"][curix], cur["代码"].to_numpy())
        label_mask = p["label_mask"] & (abs(p["y"]) >= 0.45)
        statistics = stats(xx, 0.5 * np.sign(p["y"]), label_mask)
        kbeta = solve_at(
            dict(p, x=xx), t, self.kprior, self.kpen, statistics, self.kcfg
        )
        kdelta = 0.15 * np.tanh(xx[t, curix] @ (kbeta - self.kprior) / 0.15)

        # Reconstruct historical representations inside this call, using fixed weights.
        needed = list(range(t - 21, t - 1)) + [t]
        frame, _ = build_features(
            data["Factor33_winsor"], data["Barra_factor"], data["MarketData"],
            target_dates=p["days"][needed],
        )
        latent = np.zeros((*p["mask"].shape, 42), np.float32)
        today = None
        for d in needed:
            f = frame.loc[frame["日期"].eq(p["days"][d])].sort_values("代码")
            ci = np.searchsorted(p["codes"], f["代码"].to_numpy())
            assert np.array_equal(ci, np.flatnonzero(p["mask"][d]))
            z = self.standardize(f)
            mask = np.ones((1, len(f)), bool)
            if d == t:
                nodes, ts, graph = peer.encode(self.teacher, z[None], mask)
                today = (f, nodes, ts, graph)
            else:
                c = peer.core.components(self.teacher, z[None], mask)
                sd = np.sqrt(c["cash_variance"][0])
                u = c["beta"][0] * np.exp(0.5 * c["prior_logvar"][0])[None]
                share = 1 - np.exp(c["idio_logvar"][0]) / c["cash_variance"][0]
                vals = np.column_stack([
                    c["score"][0], c["rank_score"][0], c["alpha"][0] / sd,
                    c["common_mean"][0] / sd, np.log(sd), share, u,
                ]).astype(np.float32)
                nodes = np.concatenate([np.asarray(c["hidden"][0], np.float32), vals], 1)
            q = pd.DataFrame(nodes).rank(axis=0, pct=True)
            latent[d, ci] = (q - q.mean(axis=0)).to_numpy(np.float32)

        # Centered latent ranks augment the second, separately regularized statistic.
        sx = np.concatenate([p["x"], latent], -1)[:, :, self.scfg["keep"]]
        target = pd.DataFrame(p["y"] - 0.1 * latent[:, :, 32]).where(
            p["label_mask"]
        ).rank(axis=1, pct=True)
        target = target.sub(target.mean(axis=1), axis=0).fillna(0).to_numpy(np.float32)
        ss = stats(sx, 0.5 * np.sign(target), p["label_mask"] & (abs(target) >= 0.45))
        beta = solve_at(dict(p, x=sx), t, self.sprior, self.spen, ss, self.scfg)
        sdelta = 0.15 * np.tanh(sx[t, curix] @ (beta - self.sprior) / 0.15)

        # The offline-trained head conditions its correction on today's statistical context.
        pen = np.diag([0.7 if i < 10 else 0.1 for i in keep])
        coef = solve_at(
            dict(p, x=xx), t, np.zeros(len(keep)), pen, statistics,
            dict(lookback=21, window=20, half_life=20),
        )
        state = state_panel(data, p)[t]
        context = np.concatenate([
            xx[t, curix] * coef, np.broadcast_to(state, (len(curix), 8)), xx[t, curix]
        ], -1).astype(np.float32)
        f, nodes, ts, graphs = today
        assert np.array_equal(f["代码"].to_numpy(), cur["代码"].to_numpy())
        ni, nw = graphs["factor_barra"]
        pf = peer.features(nodes, ni, nw, "factor_barra")[None]
        a = peer.apply_adapter(self.p01, pf, mask0, ts[None])[0]
        b = peer.apply_adapter(self.r1, pf, mask0, ts[None])[0]
        old = (0.475 * a.astype(float) + 0.525 * b.astype(float)).astype(np.float32)
        features = np.concatenate([pf, context[None]], -1)
        new = peer.apply_adapter(self.head, features, mask0, old)[0][0]
        delta = 0.4 * (rank(new) - rank(old[0]))

        # Preserve the upper fifth of the original signal, then reorder its complement.
        original = raw_base + kdelta
        candidate = raw_base + sdelta + delta
        ranks = rank(original)
        selected = ranks <= 0.8
        indices = np.flatnonzero(selected)
        ordered = indices[np.argsort(candidate[indices], kind="stable")]
        score = ranks.copy()
        score[ordered] = np.sort(ranks[selected])
        return pd.DataFrame(dict(
            date=stamp, code=cur["代码"].to_numpy(), score=score,
            rh50_score=rh50_score, k2_score=original, unconstrained_score=candidate,
            stat_delta=sdelta, head_delta=delta,
        ))
