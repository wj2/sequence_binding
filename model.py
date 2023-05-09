import itertools as it
import numpy as np
import sklearn.preprocessing as skp
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


def _generate_chimeric_stim(stim):
    u_feats = list(np.unique(stim[:, j]) for j in range(stim.shape[1]))
    comb_sets = set(list(it.product(*u_feats)))
    return comb_sets.difference(_make_set(stim))


def _generate_dissim_stim(stim, nm_stim):
    nm_arr = np.array(list(nm_stim))
    mask = np.zeros_like(nm_arr)
    for i in range(stim.shape[1]):
        mask[:, i] = np.isin(nm_arr[:, i], stim[:, i])
    nm_mask = np.all(~mask, axis=1)
    return _make_set(nm_arr[nm_mask])


def _generate_partial_stim(stim, nm_stim, feat_val=None):
    nm_arr = np.array(list(nm_stim))
    mask = np.zeros_like(nm_arr)
    for i in range(stim.shape[1]):
        mask[:, i] = np.isin(nm_arr[:, i], stim[:, i])
    nm_num_match = np.sum(mask, axis=1)
    nm_mask = np.logical_and(nm_num_match > 0, nm_num_match < stim.shape[1])
    if feat_val is not None:
        nm_mask = np.logical_and(nm_mask, mask[:, feat_val])
    return _make_set(nm_arr[nm_mask])


def _make_set(arr):
    return set(tuple(a) for a in arr)


def _filter_nan(ref_arr, *arrs):
    keep_mask = ~np.all(np.isnan(ref_arr), axis=(1, 2))
    new_arrs = []
    for arr in (ref_arr,) + arrs:
        new_arrs.append(arr[keep_mask])
    return new_arrs


all_nonmatch_types = (
    "dissimilar",
    "chimeric",
    "partial",
)


class DisentangledInputProbe:
    def __init__(
        self,
        n_features,
        n_values,
        t_len,
        present_len=10,
        include_fixation=True,
        include_diff_probe=True,
        n_options=2,
    ):
        """
        This class generates the input and probe stimuli, and controls the
        length of their presentation.

        Parameters
        ----------
        n_features : int
           The number of features for the stimuli.

        n_values : int
           The number of discrete values that each feature can take on.

        t_len : int
           The total length of a trial.

        present_len : int, optional
           The length of stimulus presentations. Default is 10.

        include_fixation : bool, optional
           Whether or not a "fixation point" input is included, which is one
           through the trial until the response period, at which point it becomes
           zero. Default is True.

        """
        self.include_fixation = include_fixation
        self.n_features = n_features
        self.n_values = n_values
        self.t_len = t_len
        self.rng = np.random.default_rng()
        self.stim_all = list(it.product(range(n_values), repeat=n_features))
        self.stim_all_set = set(self.stim_all)
        self.stim_enc = skp.OneHotEncoder(
            categories=(np.arange(n_values),) * n_features, sparse=False
        )
        self.stim_enc.fit(self.stim_all)

        self.stim_dims = n_values * n_features
        self.stim_inds = np.arange(self.stim_dims)
        if include_diff_probe:
            self.out_dims = n_options
            opts = tuple(range(n_options))
            self.out_enc = skp.OneHotEncoder(categories=(opts,), sparse=False)
            opts_arr = np.array(opts).reshape((len(opts), 1))
            self.out_enc.fit(opts_arr)
            self.probe_inds = self.stim_dims + np.arange(n_options * self.stim_dims)
        else:
            self.out_dims = 2
            self.out_enc = skp.OneHotEncoder(categories=((False, True),), sparse=False)
            self.out_enc.fit([[True], [False]])
            self.probe_inds = self.stim_inds
        if include_fixation:
            self.fix_ind = -1
        else:
            self.fix_ind = None

        self.inp_dims = (
            self.stim_dims + include_fixation + self.stim_dims * 2 * include_diff_probe
        )
        self.present_len = present_len
        self.include_diff_probe = include_diff_probe

    def get_rep(self, stim):
        stim_trs = []
        if len(stim.shape) == 3:
            for i in range(stim.shape[1]):
                stim_trs.append(self.stim_enc.transform(stim[:, i]))
            out = np.concatenate(stim_trs, axis=1)
        else:
            out = self.stim_enc.transform(stim)
        return out

    def sample_stim(self, n_stim):
        stim = self.rng.integers(self.n_values, size=(n_stim, self.n_features))
        rep = self.get_rep(stim)

        return stim, rep

    def get_probe_stim_options(
        self,
        stim,
        present_len=None,
        n_options=2,
        sample_chimera=False,
        option_weights=None,
        only_position=None,
        match_frac=None,
        nonmatch_types=None,
    ):
        if sample_chimera:
            nonmatch_types = ("chimeric",)
        if match_frac is not None and option_weights is None and n_options == 2:
            option_weights = np.array([match_frac, 1 - match_frac])
        n_trls, n_stim, n_feats = stim.shape
        trl_inds = np.arange(n_trls)
        seq_inds = np.arange(n_stim)
        if only_position is not None:
            match_seq_ind = np.ones(n_trls, dtype=int) * only_position
        else:
            match_seq_ind = self.rng.choice(seq_inds, size=n_trls, replace=True)
        option_match_ind = self.rng.choice(
            np.arange(n_options), size=n_trls, replace=True, p=option_weights
        )
        probe_stim = np.zeros((n_trls, n_options, n_feats))
        for ind in trl_inds:
            msi = match_seq_ind[ind]
            omi = option_match_ind[ind]
            match_stim = stim[ind, msi]
            non_match_stim = self.stim_all_set.difference(_make_set(stim[ind]))
            non_match_set = set()
            if nonmatch_types is not None:
                if "chimeric" in nonmatch_types:
                    chimeric_stim = _generate_chimeric_stim(stim[ind])
                    non_match_set = non_match_set.union(chimeric_stim)
                if "dissimilar" in nonmatch_types:
                    dissim_stim = _generate_dissim_stim(stim[ind], non_match_stim)
                    non_match_set = non_match_set.union(dissim_stim)
                if "partial" in nonmatch_types:
                    partial_stim = _generate_partial_stim(stim[ind], non_match_stim)
                    non_match_set = non_match_set.union(partial_stim)
                non_match_stim = non_match_stim.intersection(non_match_set)
            if len(non_match_stim) < n_options:
                probe_stim[ind] = np.nan
            else:
                ps = self.rng.choice(
                    list(non_match_stim), size=n_options, replace=False
                )
                probe_stim[ind] = ps
                probe_stim[ind, omi] = match_stim
        return probe_stim, option_match_ind

    def get_probe_stim(
        self, stim, present_len=None, match_frac=0.5, sample_chimera=False
    ):
        match_trials = int(np.round(match_frac * stim.shape[0]))
        inds_all = np.arange(stim.shape[0])

        probe_stim = np.zeros((stim.shape[0], 1, self.n_features))
        match_inds = self.rng.choice(inds_all, size=match_trials, replace=False)
        match_mask = np.isin(inds_all, match_inds)
        nonmatch_inds = np.array(list(set(inds_all).difference(match_inds)))
        for m_i in match_inds:
            probe_stim[m_i] = self.rng.choice(stim[m_i], axis=0, size=1)
        for nm_i in nonmatch_inds:
            tup_stim = list(tuple(x) for x in stim[nm_i])
            if sample_chimera:
                roll_first = np.roll(stim[nm_i][:, 0:1], 1, axis=0)
                new_group = np.concatenate((roll_first, stim[nm_i][:, 1:]), axis=1)
                shuff_set = list(tuple(x) for x in new_group)
                nm_stim = set(shuff_set).difference(tup_stim)
            else:
                nm_stim = self.stim_all_set.difference(tup_stim)
            if len(nm_stim) > 0:
                probe_stim[nm_i] = self.rng.choice(list(nm_stim), size=1)
            else:
                probe_stim[nm_i] = np.nan
        return probe_stim, match_mask

    def get_sample_groups(self, n_seqs, *tis, present_len=None):
        seq_arr = np.zeros((n_seqs, self.t_len, self.inp_dims))
        stim_arr = np.zeros((n_seqs, len(tis), self.n_features))
        for i, ti in enumerate(tis):
            stim, reps = self.sample_stim(n_seqs)
            seq_arr[:, ti : ti + present_len, self.stim_inds] += np.expand_dims(reps, 1)
            stim_arr[:, i] = stim
        return stim_arr, seq_arr

    def sample_seqs(self, n_seqs, *tis, probe_time=None, present_len=None, **kwargs):
        if present_len is None:
            present_len = self.present_len
        if probe_time is None:
            probe_time = self.t_len - self.present_len

        stim_arr, seq_arr = self.get_sample_groups(
            n_seqs, *tis, present_len=present_len
        )
        if self.include_diff_probe:
            p_stim, match_mask = self.get_probe_stim_options(
                stim_arr, present_len=present_len, **kwargs
            )
        else:
            p_stim, match_mask = self.get_probe_stim(
                stim_arr, present_len=present_len, **kwargs
            )
        out = _filter_nan(p_stim, stim_arr, seq_arr, match_mask)
        p_stim, stim_arr, seq_arr, match_mask = out

        p_reps = self.get_rep(np.squeeze(p_stim))
        p_end = probe_time + present_len
        seq_arr[:, probe_time:p_end, self.probe_inds] += np.expand_dims(p_reps, 1)

        if self.include_fixation:
            fixation = np.ones((seq_arr.shape[0], seq_arr.shape[1]))
            fixation[:, probe_time:] = 0
            seq_arr[..., self.fix_ind] = fixation

        target_img = self.out_enc.transform(np.expand_dims(match_mask, 1))
        target_arr = np.zeros((stim_arr.shape[0], self.t_len, target_img.shape[1]))
        target_arr[:, probe_time:] = np.expand_dims(target_img, 1)
        return (stim_arr, p_stim, match_mask), seq_arr, target_arr


class SequenceRNN:
    def __init__(
        self,
        probe,
        n_units,
        target_window=10,
        use_early_stopping=True,
        early_stopping_field="val_loss",
        **kwargs
    ):
        """
        This class generates the input and probe stimuli, and controls the
        length of their presentation.

        Parameters
        ----------
        probe : InputProbe
           The probe object that the network will use to generate input.

        n_units : int
           The number of units in the recurrent part of the network.

        target_window : int, optional
           The length of the window in which the network will be trained to
           output the target value. Default is 10.

        use_early_stopping : bool, optional
           Whether or not to use early stopping during training. Default is True.

        early_stopping_field : str, optional
           Field to use to decide whether or not to stop training early. Default
           is 'val_class_branch_loss'.

        All other keyword parameters are ignored.
        """
        self.probe = probe
        self.input_units = probe.inp_dims
        self.output_units = probe.out_dims
        self.n_units = n_units
        self.t_len = probe.t_len
        self.target_window = target_window

        out = self.make_model(
            self.t_len,
            self.n_units,
            self.input_units,
            self.output_units,
            target_window,
            **kwargs
        )
        self.model, self.rnn = out
        self.compiled = False
        self.use_early_stopping = True
        self.early_stopping_field = early_stopping_field

    def _compile(self, optimizer=None, loss=tf.losses.BinaryCrossentropy()):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_dynamics(self, inp):
        return self.rnn(inp)

    def get_output(self, inp):
        return self.model(inp)

    def sample_trajectories(self, n_traj, *ts, **kwargs):
        out = self.probe.sample_seqs(n_traj, *ts, **kwargs)
        stim_info, inp_arr, targ_arr = out
        dyn = self.get_dynamics(inp_arr)
        resp = self.get_output(inp_arr)
        return stim_info, dyn, resp

    def compute_percent_correct(self, *ts, n_samps=1000, shuffle=False, **kwargs):
        _, inp_arr, targ_arr = self.probe.sample_seqs(n_samps, *ts, **kwargs)

        out_m = self.model(inp_arr)
        resp_mask = out_m[:, -1] > 0.5
        targ_mask = targ_arr[:, -1]
        if shuffle:
            self.probe.rng.shuffle(targ_mask)
        pc = np.nanmean(resp_mask == targ_mask)
        return pc

    def make_model(
        self, t_len, n_units, n_inp, n_out, target_win, act_func=tf.nn.relu, **kwargs
    ):
        inp = tfkl.Input(shape=(t_len, n_inp))
        rnn = tfkl.SimpleRNN(n_units, activation=act_func, return_sequences=True)
        full_mask = np.zeros((t_len, n_out), dtype=bool)
        full_mask[-target_win:] = True
        out = tfkl.Dense(n_out, activation=tf.nn.sigmoid)

        full_layers = [inp, rnn, out]
        model = tfk.Sequential(full_layers)
        rnn_layers = [inp, rnn]
        model_rnn = tfk.Sequential(rnn_layers)

        return model, model_rnn

    def fit(
        self,
        *ts,
        n_samples=10000,
        match=0.5,
        batch_size=200,
        n_epochs=10,
        val_samples=1000,
        **kwargs
    ):
        """
        This method controls the fitting of the network to a particular set of
        stimulus presentation times.

        Parameters
        ----------
        *ts : list of ints
           The presentation times of the stimuli.

        n_samples : int, optional
           The number of samples used for training.

        match : float
           The fraction of match trials used in training.

        batch_size : int, optional
           The batch size used for training.

        n_epochs : int, optional
           The number of epochs the network is trained for.

        val_samples : int, optional
           The number of samples in the validation set.

        All keyword arguments are passed to the TensorFlow fit method.
        """

        if not self.compiled:
            self._compile()
        if self.use_early_stopping:
            cb = tfk.callbacks.EarlyStopping(
                monitor=self.early_stopping_field, mode="min", patience=5
            )
            curr_cb = kwargs.get("callbacks", [])
            curr_cb.append(cb)
            kwargs["callbacks"] = curr_cb

        out = self.probe.sample_seqs(n_samples, *ts, match_frac=match)
        stims, inp_arr, targ_arr = out

        out = self.probe.sample_seqs(val_samples, *ts, match_frac=match)
        _, val_inp, val_targ = out
        val_set = (val_inp, val_targ)

        hist = self.model.fit(
            x=inp_arr,
            y=targ_arr,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=val_set,
            **kwargs
        )
        return hist
