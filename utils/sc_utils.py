import itertools
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import math
from scipy import optimize
from tqdm import tqdm


def gen_chunks(lst, num_chunks):
    """Yield num_chunks chunks from lst."""
    n = int(np.ceil(len(lst) / num_chunks))
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rmse(weights: np.array, control_vals: np.array, target: np.array):
    return np.sqrt(((control_vals @ weights - target) ** 2).mean())


@ray.remote(num_returns=1)
def eval_rmse(df: pd.DataFrame, combo_chunk: List[Tuple[str]], weight_chunk: List[np.array], target: np.array) -> List[float]:
    test_val_errs = []
    for combo, weights in zip(combo_chunk, weight_chunk):
        c_vals = df.loc[df['state'].isin(combo)].pivot_table(values='daily_packs', index='year', columns='state').values
        test_val_errs.append(rmse(weights=weights, control_vals=c_vals, target=target))
    
    return test_val_errs


def gen_oos_err(res_df: pd.DataFrame, oos_yrs: List[int], num_chunks: int, orig_df: pd.DataFrame, target_state: str = 'CA') -> List[float]:
    res_df = res_df.copy()
    
    test_yr = orig_df.loc[orig_df['year'].isin(oos_yrs)].copy()
    test_target = test_yr.loc[test_yr['state'] == target_state]['daily_packs'].values

    combo_chunks = gen_chunks(list(res_df['combo']), num_chunks)
    weight_chunks = gen_chunks(list(res_df['weights']), num_chunks)
    
    futures = [eval_rmse.remote(df=test_yr, combo_chunk=c, weight_chunk=w, target=test_target) for c,w in zip(combo_chunks, weight_chunks)]
    errs = list(itertools.chain.from_iterable(ray.get(futures)))
                                   
    return errs


def reg_obj(weights: np.array, control_vals: np.array, target: np.array, penalty: float):
    return rmse(weights=weights, control_vals=control_vals, target=target) + penalty * np.linalg.norm(weights)


@ray.remote(num_returns=1)
def find_weights(control_val_chunk: List[np.array], target: np.array, penalty: float = 1) -> Tuple[List[np.array]]:    
    weights = []
    objs = []
    is_rmse = []
    for control_vals in control_val_chunk:
        res = optimize.minimize(
            reg_obj, 
            args=(control_vals, target, penalty),
            x0=np.ones(control_vals.shape[1]) / control_vals.shape[1],  # since this guess is even, res should be the min l2-norm weights
            method='SLSQP', 
            bounds=tuple((0,1) for x in range(control_vals.shape[1])),  # require weights to be nonnegative
            constraints=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})  # require weights to sum to 1
        )
        
        weights.append(res.x)
        objs.append(res.fun)
        is_rmse.append(np.sqrt(((control_vals @ res.x - target) ** 2).mean()))
        
    return (weights, objs, is_rmse)


@ray.remote(num_returns=1)
def get_combo_mats(df: pd.DataFrame, combos: List[Tuple[str]]) -> Dict[tuple, np.array]:    
    control_mat = []
    for combo in combos:
        control_mat.append(df.loc[df['state'].isin(combo)].pivot_table(values='daily_packs', index='year', columns='state').values)
            
    return dict(zip(combos, control_mat))


def run_generalized(
    orig_df: pd.DataFrame, 
    num_lags: int, 
    num_chunks: int, 
    control_set: List[Tuple[str]], 
    target_state: str = 'CA',
    gap: int = 0, 
    penalty: float = 1, 
    verbose: bool = True
) -> pd.DataFrame:
    
    train_yrs = [1989 - gap - l for l in range(1, num_lags + 1)]
    test_yrs = [1989 - g for g in range(1, gap + 1)]
    
    if verbose:
        print('target state:', target_state)
        print('training on:', train_yrs)
        print('testing on:', test_yrs)

    df = orig_df.loc[orig_df['year'].isin(train_yrs)].copy()
    target = df.loc[df['state'] == target_state].sort_values('year')['daily_packs'].values
    
    combo_chunks = gen_chunks(control_set, num_chunks)
    
    combo_futures = [get_combo_mats.remote(df=df, combos=chunk) for chunk in combo_chunks]
    combo_res = ray.get(combo_futures)
    
    # combine all the dicts into 1
    combos = combo_res[0]
    for d in combo_res[1:]:
        combos.update(d)
    
    if verbose:
        print('reconstructed control matrices')

    # chunk the matrices
    mat_chunks = gen_chunks(list(combos.values()), num_chunks)
    
    futures = [find_weights.remote(control_val_chunk=chunk, target=target, penalty=penalty) for chunk in mat_chunks]
    res = list(itertools.chain(ray.get(futures)))
    
    if verbose:
        print('obtained weights')
    
    res_w = []
    res_o = []
    res_i = []
    for r in res:
        res_w.append(r[0])
        res_o.append(r[1])
        res_i.append(r[2])
        
    res_w = list(itertools.chain.from_iterable(res_w))
    res_o = list(itertools.chain.from_iterable(res_o))
    res_i = list(itertools.chain.from_iterable(res_i))
    
    res_w_dict = dict()
    res_i_dict = dict()
    res_o_dict = dict()
    for c, w, i, o in zip(combos.keys(), res_w, res_i, res_o):
        if not np.isnan(w.max()):
            assert len(c) == len(w), f'{c}, {w}'
            res_w_dict[c] = w
            res_i_dict[c] = i
            res_o_dict[c] = o
    
    res_df = pd.DataFrame({"combo": res_w_dict.keys(), "weights": res_w_dict.values(), "obj_val": res_o_dict.values(), "err_is": res_i_dict.values()})
    res_df['l2_norm'] = res_df['weights'].apply(lambda x: np.sqrt((x ** 2).sum()))
    res_df['err_oos'] = gen_oos_err(orig_df=orig_df, res_df=res_df, oos_yrs=test_yrs, num_chunks=num_chunks, target_state=target_state)
    
    return res_df


def gen_series(df, num_states, target_state, num_lags=3, gap=2, seed=10):
    np.random.seed(seed)
    sub_states = df.loc[df['state'] != target_state]['state'].drop_duplicates().sample(n=num_states,
                                                                                       replace=False).sort_values().values

    res_dfs = []
    for n in tqdm(range(1, num_states + 1)):
        res_df = run_generalized(control_set=list(itertools.combinations(sub_states, n)), orig_df=df,
                                 target_state=target_state, num_lags=num_lags, gap=gap, num_chunks=48, penalty=0,
                                 verbose=False)
        res_df['n'] = n
        res_dfs.append(res_df)

    return pd.concat(res_dfs).reset_index().drop(columns=['index']).copy()


def plot_series(res, target_state, avg=True, num_lags=3):
    if avg:
        stats = res.groupby('n').agg({'err_oos': ['mean', 'std', 'count']}).droplevel(level=0, axis=1).reset_index()
        stats['std_mean'] = stats['std'] / np.sqrt(stats['count'])
        fig = stats.plot(kind='scatter', x='n', y='mean', yerr='std_mean', xlabel='# Control States',
                         ylabel='Mean RMSE', title=f'Target {target_state}')
    else:
        alt_stat = pd.concat([
            res.loc[res.groupby('n').err_is.idxmin()].head(num_lags),  # the first n are non-interpolating
            res.loc[res.groupby('n').l2_norm.idxmin()].iloc[num_lags:]
            # the remaining are interpolating, so go by L2 norm
        ]).reset_index().drop(columns=['index']).copy()

        fig, ax = plt.subplots()
        alt_stat.plot(kind='line', x='n', y='err_oos', xlabel='# Control States', ylabel='RMSE',
                      title=f'Target {target_state}', ax=ax)
        ax.axvline(num_lags, color="red", linestyle="dashed")
    return fig


def run_series(df, num_states, target_state, num_lags=3, gap=2, seed=10, plot=False):
    res = gen_series(df=df, num_states=num_states, target_state=target_state, num_lags=num_lags, gap=gap, seed=seed)
    if plot:
        fig_avg = plot_series(res, target_state=target_state)
        fig_best = plot_series(res, target_state=target_state, num_lags=num_lags, avg=False)
        return res, fig_avg, fig_best
    else:
        return res

    
def get_random_subsets(parent_set, n_units, n_samples, target='CA', seed=0):
    np.random.seed(seed)
    
    total_comb = math.comb(len(parent_set), n_units)
    if total_comb < n_samples:
        n_samples = total_comb
    
    samples = []
    for _ in range(n_samples):
        choice = tuple(sorted(np.asarray(parent_set)[
            np.random.choice(len(parent_set), replace=False, size=n_units)
        ]))
        
        while choice in samples:
            choice = tuple(sorted(np.asarray(parent_set)[
                np.random.choice(len(parent_set), replace=False, size=n_units)
            ]))
        
        samples.append(choice)
    
    return samples


def run_series_random(df, num_states, num_samples, target_state='CA', num_lags=10, gap=2, seed=0, num_chunks=48):
    states = df.loc[df['state'] != target_state]['state'].drop_duplicates().sort_values().values
    
    res_dfs = []
    for n in tqdm(range(1, num_states+1)):
        samples = get_random_subsets(parent_set=states, n_units=n, n_samples=num_samples, target=target_state, seed=seed)
        
        res_df = run_generalized(
            control_set=samples, 
            orig_df=df, 
            target_state=target_state, 
            num_lags=num_lags, 
            gap=gap, 
            num_chunks=num_chunks, 
            penalty=0, 
            verbose=False
        )
        
        res_df['n'] = n
        res_dfs.append(res_df)
        
    return pd.concat(res_dfs).reset_index().drop(columns=['index']).copy()
