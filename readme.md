
# Code for modeling sequence element recall with an RNN

To get started, run:
``` 
import sequence_binding.model as sbm

t_len = 55
k = 2
n_v = 3
n_units = 50

dis = sbm.DisentangledInputProbe(k, n_v, t_len)
seq = sbm.SequenceRNN(dis, n_units)

ts = (10, 30)

seq.fit(*ts, n_epochs=200)
```

This will train the model `seq` using stimuli presented over 55 time steps
at both timestep 10 and 30. 

To quantify the error rate, run:
```
print(seq.compute_percent_correct(*ts, n_samps=10000, match_frac=0))
```

To plot some trajectories from within the model, run:
```
import matplotlib.pyplot as plt

stim_info, out_rnn, out_m = seq.sample_trajectories(10, *ts, match_frac=0)

f, (ax1, ax2) = plt.subplots(1, 2)

_ = ax1.plot(out_rnn[0])
_ = ax2.plot(out_m[0])
```
