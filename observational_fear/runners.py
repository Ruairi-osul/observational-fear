from tqdm.notebook import tqdm
import pandas as pd
from typing import List, Tuple

class ExperimentRunner:
    def __init__(self, decoder, n_shuffles):
        self.decoder = decoder
        self.n_shuffles = n_shuffles
    
    
    def _run_single(self, frame_sets: List[Tuple[pd.DataFrame, pd.DataFrame]]):
        scores = self.decoder.fit_models(frame_sets)
        return scores
    
    def _shuffle_neg(self, trace_handler):
        positive = trace_handler.aligned_wide
        negative = trace_handler.trial_pivot(trace_handler.generate_rotated())
        return positive, negative

    def _shuffle_both(self, trace_handler):
        positive = trace_handler.trial_pivot(trace_handler.generate_rotated())
        negative = trace_handler.trial_pivot(trace_handler.generate_rotated())
        return positive, negative
    
    def _shuffle_single(self, trace_handler):
        return trace_handler.trial_pivot(trace_handler.generate_rotated())

    
    def run_single(self, trace_handler):
        pos_v_shuffles = []
        for i in tqdm(range(self.n_shuffles), desc="Positive Versus Shuffle"):
            positive, negative = self._shuffle_neg(trace_handler)
            scores = self._run_single([(positive, negative)]).assign(shuffle=i)
            pos_v_shuffles.append(scores)
        df_pos = pd.concat(pos_v_shuffles).assign(config="Positive v Shuffles")
        shuffles_v_shuffles = []
        for i in tqdm(range(self.n_shuffles), desc="Shuffle Versus Shuffle"):
            positive, negative = self._shuffle_both(trace_handler)
            scores = self._run_single([(positive, negative)]).assign(shuffle=i)
            shuffles_v_shuffles.append(scores)
        df_shuf = pd.concat(shuffles_v_shuffles).assign(config="Shuffles v Shuffles")
        return pd.concat([df_pos, df_shuf])
    
    def run_combined(self, trace_handler1, trace_handler2):
        positive = trace_handler1.aligned_wide
        negative = trace_handler2.aligned_wide
        df_pos = self._run_single(positive, negative).assign(shuffle=1, config="Positive v Positive")
        
        # shuffle2 versus shuffle1
        shuffles_v_shuffles = []
        for i in tqdm(range(self.n_shuffles), desc="Shuffle Versus Shuffle"):
            positive = self._shuffle_single(trace_handler1)
            negative = self._shuffle_single(trace_handler2)
            scores = self._run_single(positive, negative).assign(shuffle=i)
            shuffles_v_shuffles.append(scores)
        df_shuf = pd.concat(shuffles_v_shuffles).assign(config="Shuffles v Shuffles")
        return pd.concat([df_pos, df_shuf])
    

    def run_milti(self, trace_handler1, trace_handler2):
        pos_frames = []
        for i in tqdm(range(self.n_shuffles), desc="Positive Included"):
            positive1, negative1 = self._shuffle_neg(trace_handler1)
            positive2, negative2 = self._shuffle_neg(trace_handler2)
            scores = self._run_single([(positive1, negative1), (positive2, negative2)]).assign(shuffle=i)
            pos_frames.append(scores)
        df_pos = pd.concat(pos_frames).assign(config="Pos1 v Neg1 v Pos2 v Neg2")
        neg_frames = []
        for i in tqdm(range(self.n_shuffles), desc="Negative Only"):
            positive1, negative1 = self._shuffle_both(trace_handler1)
            positive2, negative2 = self._shuffle_both(trace_handler2)
            scores = self._run_single([(positive1, negative1), (positive2, negative2)]).assign(shuffle=i)
            neg_frames.append(scores)
        df_neg = pd.concat(neg_frames).assign(config="Neg1 v Neg11 v Neg2 v Neg22")
        return pd.concat([df_pos, df_neg])

