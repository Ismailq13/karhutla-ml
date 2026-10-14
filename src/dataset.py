# File: src/dataset.py — pembagi waktu dan pembuat dataset tf.data
import numpy as np, tensorflow as tf

def make_split(dates, T=7, LEAD=1):
    # Split kronologis berdasarkan TAHUN TARGET, bukan acak (acak = kebocoran waktu).
    t_all = np.arange(T - 1, len(dates) - LEAD)
    tgt_year = dates[t_all + LEAD].year
    return (t_all[tgt_year <= 2022], t_all[tgt_year == 2023], t_all[tgt_year == 2024])

def make_ds(idx, dyn, sta, Y, MASK, T=7, LEAD=1, batch=2, shuffle=False):
    H, W = MASK.shape
    C = dyn.shape[-1] + sta.shape[-1]
    def gen():
        order = np.random.permutation(idx) if shuffle else idx
        for t in order:
            # x: 7 hari input dinamis + peta statis diulang 7 kali
            x = np.concatenate([dyn[t - T + 1:t + 1], np.repeat(sta[None], T, 0)], -1)
            y = np.stack([Y[t + LEAD], MASK], -1)      # kanal 0 = label, kanal 1 = mask gambut
            yield x.astype('float32'), y.astype('float32')
    sig = (tf.TensorSpec((T, H, W, C), tf.float32), tf.TensorSpec((H, W, 2), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch).prefetch(2)
