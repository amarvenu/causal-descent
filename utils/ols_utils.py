import itertools
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import random
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from statsmodels.tools.tools import add_constant
from tqdm import tqdm


def plot_preview(errs: np.array, dims: List[int], train: np.array, name: str, plot: bool, loss_type: str, verbose: bool = True):
    compare = pd.DataFrame({"err": errs}, index=np.array(dims) / train.shape[0])

    fig, ax = plt.subplots(figsize=(9,6))
    
    compare.plot(ax=ax, ylim=(0, max(1.5, int(compare['err'].max()/2.5))))

    ax.axvline(1, color="red", linestyle="dashed")
    ax.axhline(train.std(), color="green", linestyle="dashed")
    
    if loss_type == 'rmse':
        best_val = compare['err'].min()
    else:
        best_val = compare['err'].max()
    
    ax.axhline(best_val, color="orange", linestyle="dotted", linewidth=4)
    
    if verbose:
        print(best_val, train.std())

    ax.set_title(fr'Out of Sample {loss_type.upper()} vs. $\gamma$ ({name})')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(f'Out of Sample {loss_type.upper()}')

    if plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig 


def calc_pred(X_train: np.array, y_train: np.array, X_test: np.array, ridge_eps: float = None, ret_beta: bool = False):
    if ridge_eps is not None:
        square = X_train.T @ X_train
        ridge = ridge_eps * np.identity(square.shape[0])
        beta = np.linalg.inv(square + ridge) @ X_train.T @ y_train
        pred = X_test @ beta
    else:
        beta = np.linalg.pinv(X_train) @ y_train
        pred = X_test @ np.linalg.pinv(X_train) @ y_train
        
    if ret_beta:
        return pred, beta
    else:
        return pred
    

def calc_loss(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, ridge_eps: float = None, loss_type: str = 'rmse', ret_beta: bool = False) -> float:
    res = calc_pred(X_train=X_train, y_train=y_train, X_test=X_test, ridge_eps=ridge_eps, ret_beta=ret_beta)
    if ret_beta:
        pred = res[0].copy()
        beta = res[1].copy()
    else:
        pred = res.copy()
    
    if loss_type == 'rmse':
        err = np.sqrt(mean_squared_error(y_test, pred)) 
    elif loss_type in ['f1', 'accuracy']:
        pred[pred < 0] = 0
        pred[pred > 1] = 1
        
        if loss_type == 'f1':
            err = f1_score(y_test, np.round(pred))
        else:
            err = accuracy_score(y_test, np.round(pred))
    else:
        raise ValueError('unsupported loss type')
        
    if ret_beta:
        return err, beta
    else:
        return err


def get_random_subset_X(seed: int, n_cols: int, k: int) -> np.array:
    np.random.seed(seed)
    return np.random.choice(list(range(n_cols)), k, replace=False)


@ray.remote(num_returns=1)
def eval_rf(
    k: int, 
    X_file: str, 
    y_file: str, 
    fix_seed: bool, 
    benchmark_file: str = None, 
    ridge_eps: float = None, 
    loss_type: str = 'rmse',
    feature_order: np.array = None,
    intercept: bool = False,
    ret_beta: bool = False,
    ret_cols: bool = False
) -> float:
    ys = np.load(y_file)
    Xs = np.load(X_file)
    
    if intercept:
        k -= 1  ## replace 1 col with intercept
    
    if fix_seed:
        seed = k
    else:
        seed = None
    
    if benchmark_file is None:
        if feature_order is None:            
            col_idxs = get_random_subset_X(seed=seed, n_cols=Xs['train'].shape[1], k=k)
        else:
            col_idxs = feature_order[:k]  # choose the first k of the random order
        
        X_train = Xs['train'][:, col_idxs]
        X_test = Xs['test'][:, col_idxs]
    else:
        benchmark = np.load(benchmark_file)

        if feature_order is None:
            col_idxs = get_random_subset_X(seed=seed, n_cols=Xs['train'].shape[1], k=int(k - benchmark['train'].shape[1]))
        else:
            col_idxs = feature_order[:int(k - benchmark['train'].shape[1])]

        X_train = np.concatenate([benchmark['train'], Xs['train'][:, col_idxs]], axis=1)
        X_test = np.concatenate([benchmark['test'], Xs['test'][:, col_idxs]], axis=1)
    
    if intercept:
        X_train = add_constant(X_train)
        X_test = add_constant(X_test)
    
    loss = calc_loss(X_train=X_train, y_train=ys['train'], X_test=X_test, y_test=ys['test'], ridge_eps=ridge_eps, loss_type=loss_type, ret_beta=ret_beta)
    
    if ret_cols:
        if ret_beta:
            return loss[0], loss[1], col_idxs
        else:
            return loss, col_idxs
    else:
        return loss



def run_rf(
    X_file: str, 
    y_file: str,
    name: str = '',
    k_min: int = 10, 
    k_max: int = None, 
    k_step: int = 10, 
    seed: int = 10, 
    plot: bool = True, 
    verbose: bool = True,
    benchmark_file: str = None,
    ridge_eps: float = None,
    loss_type: str = 'rmse',
    nest: bool = False,
    intercept: bool = False,
    ret_beta: bool = False,
    ret_cols: bool = False
):

    if k_max is None:
        k_max = np.load(X_file)['train'].shape[1]
        if benchmark_file is not None:
            k_max += np.load(benchmark_file)['train'].shape[1]
        if intercept:
            k_max += 1  # account for added intercept col
    
    ks = list(range(k_min, k_max, k_step))
    
    if k_max not in ks:
        ks += [k_max]
    
    k_peak = np.load(X_file)['train'].shape[0]
    if intercept:
        k_peak += 1  # account for added intercept col
    if k_peak not in ks:
        low_ks = [k for k in ks if k < k_peak]
        high_ks = [k for k in ks if k > k_peak]
        ks = low_ks + [k_peak] + high_ks
    
    if nest:
        if seed:
            np.random.seed(seed)
        feature_order = np.random.choice(range(np.load(X_file)['train'].shape[1]), size=np.load(X_file)['train'].shape[1], replace=False)
    else:
        feature_order = None
    
    err_futures = [eval_rf.remote(
        k=k, 
        X_file=X_file, 
        y_file=y_file, 
        fix_seed=bool(seed), 
        benchmark_file=benchmark_file, 
        ridge_eps=ridge_eps, 
        loss_type=loss_type,
        feature_order=feature_order,
        intercept=intercept,
        ret_beta=ret_beta,
        ret_cols=ret_cols
    ) for k in ks]
    
    res = ray.get(err_futures)
    
    if ret_beta:
        err = [r[0] for r in res]
        beta = [r[1] for r in res]
        if ret_cols:
            col_idxs = [r[2] for r in res]
    else:
        if ret_cols:
            err = [r[0] for r in res]
            col_idxs = [r[1] for r in res]
        else:
            err = res
    
    fig = plot_preview(errs=err, dims=ks, train=np.load(y_file)['train'], name=name, plot=plot, verbose=verbose, loss_type=loss_type)
        
    if ret_beta:
        if ret_cols:
            return err, fig, ks, beta, col_idxs
        else:
            return err, fig, ks, beta
    else:
        if ret_cols: 
            return err, fig, ks, col_idxs
        else:
            return err, fig, ks

    
def plot_err(
    errs: np.array, 
    dims: List[int], 
    ys: np.array, 
    benchmark_Xs: np.array, 
    name: str, 
    y_bounds: Tuple[float, float]=(0,100), 
    x_bounds: Tuple[float, float]=(0,9000),
    loss_type: str = 'rmse',
    yerrs: np.array = None,
    intercept: bool = False
):
    
    y_train = ys['train']
    y_test = ys['test']
    
    benchmark_X_train = benchmark_Xs['train']
    benchmark_X_test = benchmark_Xs['test']
    
    if intercept:
        benchmark_X_train = add_constant(benchmark_X_train)
        benchmark_X_test = add_constant(benchmark_X_test)

    
    compare = pd.DataFrame({"err": errs}, index=np.array(dims))

    fig, ax = plt.subplots(figsize=(9,6))
    
    compare.plot(ax=ax, ylim=y_bounds, xlim=x_bounds)
    
    ols_benchmark = benchmark_X_test @ np.linalg.pinv(benchmark_X_train) @ y_train
    
    if loss_type == 'rmse':
        best_val = compare['err'].min() 
        mean_benchmark_err = np.sqrt(mean_squared_error(y_test, [y_train.mean()] * y_test.shape[0]))
        ols_benchmark_err = np.sqrt(mean_squared_error(y_test, ols_benchmark))
    else:
        best_val = compare['err'].max()
        
        ols_benchmark[ols_benchmark < 0] = 0
        ols_benchmark[ols_benchmark > 1] = 1
        
        if loss_type == 'f1':
            mean_benchmark_err = f1_score(y_test, [np.round(y_train.mean())] * y_test.shape[0])
            ols_benchmark_err = f1_score(y_test, np.round(ols_benchmark))
        elif loss_type == 'accuracy':
            mean_benchmark_err = accuracy_score(y_test, [np.round(y_train.mean())] * y_test.shape[0])
            ols_benchmark_err = accuracy_score(y_test, np.round(ols_benchmark))
    
    # line showing interpolation threshold
    ax.axvline(benchmark_X_train.shape[0], color="red", linestyle="dashed")
    
    # line showing best performance value
    ax.axhline(best_val, color="orange", linestyle="dotted", linewidth=4)
    
    # line where we just guess the in-sample mean
    ax.axhline(mean_benchmark_err, color="black", linestyle="dotted", linewidth=4)
    
    # line where we just use vanilla OLS model fitted on the default feature set
    ax.axhline(ols_benchmark_err, color="blue", linestyle="dotted", linewidth=4)
    
    if yerrs is not None:
        ax.fill_between(compare.index, compare['err'] - yerrs, compare['err'] + yerrs)

    print('Best Metric:', best_val)
    print('Mean Guess Metric:', mean_benchmark_err)
    print('Normal OLS Metric:', ols_benchmark_err)

    ax.set_title(fr'Out of Sample {loss_type.upper()} vs. k({name})')
    ax.set_xlabel('k')
    ax.set_ylabel(f'Out of Sample {loss_type.upper()}')

    plt.show()
    
    return fig


def get_n_qbins(series, target):
    n = target
    num_bins = pd.qcut(series, q=n, duplicates='drop', labels=False).nunique()
    while num_bins < target:
        n += 1
        num_bins = pd.qcut(series, q=n, duplicates='drop', labels=False).nunique()

    print(f'Using {n} quantiles to achieve {num_bins} bins for {series.name}')
    return pd.qcut(series, q=n, duplicates='drop', labels=False)


def construct_interactions(df, order=1):
    # create dummies out of the discrete variables
    age_dummies = pd.get_dummies(df['age'], prefix='age')
    education_dummies = pd.get_dummies(df['education'], prefix='educ')

    re74_discrete = get_n_qbins(df['re74'], 50) + 1
    re74_dummies = pd.get_dummies(re74_discrete, prefix='re74_qbin')

    re75_discrete = get_n_qbins(df['re75'], 50) + 1
    re75_dummies = pd.get_dummies(re75_discrete, prefix='re75_qbin')

    # black, hispanic are mutually exclusive, so should not be interacted
    race = df[['black', 'hispanic']].copy()
    # now, compile all of the useful columns into one big df
    raw = pd.concat(
        [df[['test', 're78', 'married', 'nodegree']], race, age_dummies, education_dummies, re74_dummies, re75_dummies],
        axis=1)

    # get all of the combos that we can interact
    names = [['married'], ['nodegree'], list(race.columns), list(age_dummies.columns), list(education_dummies.columns),
             list(re74_dummies.columns), list(re75_dummies.columns)]

    col_combos = []
    for pair in itertools.combinations(names, 2):
        col_combos += list(itertools.product(pair[0], pair[1]))

    cols = []
    for i in range(1, order + 1):
        for combo in col_combos:
            cols.append(pd.Series(data=(raw[combo[0]] * raw[combo[1]]) ** i, name=f"({combo[0]} + {combo[1]}) ** {i}"))

    mat = pd.concat(cols, axis=1)
    mat['label'] = raw['re78']
    mat['test'] = raw['test']

    return mat


def write_train_test(df, n_train, outname, benchmark_df, noisy=True, seed=174, exptest=True):
    np.random.seed(seed)

    if noisy:
        if exptest:
            noise = np.random.normal(loc=0, scale=0.02, size=(df.shape[0], df.shape[1] - 2))
            noisy_df = (df.drop(columns=['label', 'test']) + noise).copy()
            noisy_df['test'] = df['test'].values
        else:
            noise = np.random.normal(loc=0, scale=0.02, size=(df.shape[0], df.shape[1] - 1))
            noisy_df = (df.drop(columns=['label']) + noise).copy()

        noisy_df['label'] = df['label'].values
        df = noisy_df.copy()

    if exptest:
        row_idxs = np.random.choice(range(df.loc[df['test'] == 0].shape[0]), size=n_train, replace=False)

        train = df.loc[df['test'] == 0].iloc[row_idxs, :].copy()
        test = df.loc[df['test'] == 1].copy()

        np.savez_compressed(f'{outname}_features.npz', train=train.drop(columns=['label', 'test']).values,
                            test=test.drop(columns=['label', 'test']).values)
    else:
        row_idxs = np.random.choice(range(df.shape[0]), size=n_train, replace=False)
        train = df.iloc[row_idxs, :].copy()
        test = df.iloc[~df.index.isin(train.index), :].copy()

        np.savez_compressed(f'{outname}_features.npz', train=train.drop(columns=['label']).values,
                            test=test.drop(columns=['label']).values)

    np.savez_compressed(f'{outname}_label.npz', train=train['label'].values,
                        test=test['label'].values)

    print('# Train:', train.shape[0])
    print('# Test:', test.shape[0])

    benchmark = benchmark_df[
        ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75', 're78']].rename(
        {"re78": "label"}, axis=1).copy()
    if exptest:
        benchmark['test'] = benchmark_df['test'].values
        benchmark_train = benchmark.loc[benchmark['test'] == 0].iloc[row_idxs, :].drop(columns=['test']).copy()
        benchmark_test = benchmark.loc[benchmark['test'] == 1].drop(columns=['test']).copy()
    else:
        benchmark_train = benchmark.iloc[row_idxs, :].copy()
        benchmark_test = benchmark.iloc[~benchmark.index.isin(benchmark_train.index), :].copy()

    np.savez_compressed(f'{outname}_benchmark_label.npz', train=benchmark_train['label'].values,
                        test=benchmark_test['label'].values)
    np.savez_compressed(f'{outname}_benchmark_features.npz',
                        train=benchmark_train.drop(columns=['label']).values,
                        test=benchmark_test.drop(columns=['label']).values)

    
def pad_betas(betas, col_idxs, k_max):
    padded_betas = []
    for b in betas:
        if b.shape[0] < k_max:
            b = np.pad(b, (0, k_max - b.shape[0]))
        padded_betas.append(b)
    
    return np.stack(padded_betas)
    

def run_ate(
    X_file: str, 
    y_file: str, 
    name='cps_control', 
    k_step: int = 50, 
    ms: List[int] = [5, 10, 20, 50, 100],
    n_draws: int = 1000,
    seed: int = 0,
    n_runs: int = 5
) -> Tuple[pd.DataFrame, Dict[int, float], Dict[int, float]]:
    
    test_X = np.load(X_file)['test']
    test_y = np.load(y_file)['test']

    samples = {1: list(range(test_X.shape[0]))}
    random.seed(seed)
    for m in tqdm(ms):
        samples[m] = []
        for i in range(n_draws):
            draw = random.sample(range(test_X.shape[0]), m) 
            while draw in samples[m]:
                draw = random.sample(range(test_X.shape[0]), m)
            samples[m].append(draw)
        
    res = []
    beta_norm_dict = {}
    for seed in tqdm(range(n_runs)):
        test_nest = run_rf(
            X_file=X_file, 
            y_file=y_file, 
            name=name, 
            k_step=k_step,
            nest=True, 
            plot=False,
            verbose=False,
            seed=seed,
            intercept=True,
            ret_beta=True,
            ret_cols=True
        )

        col_idxs = test_nest[-1]
        betas = test_nest[-2]

        padded_betas = pad_betas(betas, col_idxs, max(test_nest[2]))

        beta_norm_dict[seed] = np.linalg.norm(padded_betas, axis=1)

        products = padded_betas @ add_constant(test_X[:, col_idxs[-1]]).T   ## take product with correctly ordered columns

        err_curve = dict()
        for m in samples.keys():  # choose a subset cardinality
            err_avgs = []
            for combo in samples[m]:
                if m == 1:
                    combo = [combo]
                avg_pred = products[:, combo].mean(axis=1)  # avg predicted outcome
                avg_target = test_y[combo].mean()  # avg true outcome
                err_avgs.append((avg_pred - avg_target) ** 2)  # squared diff in avg pred

            err_curve[m] = np.stack(err_avgs).mean(axis=0)

        tmp = pd.DataFrame(err_curve)
        tmp['k'] = test_nest[2]
        tmp['seed'] = seed

        res.append(tmp)

    res = pd.concat(res)

    grouped = (res.groupby('k').mean()).reset_index().drop(columns=['seed'])
    grouped.columns = ['k'] + [f'n_{m}' for m in [1] + ms]

    for col in grouped.columns:
        if col.startswith('n_'):
            grouped[col] = np.sqrt(grouped[col])  
            
    benchmark_Xs = np.load(X_file.replace('features', 'benchmark_features'))
    benchmark_ys = np.load(y_file.replace('label', 'benchmark_label'))

    benchmark_X_train = benchmark_Xs['train']
    benchmark_X_test = benchmark_Xs['test']

    benchmark_y_train = benchmark_ys['train']
    benchmark_y_test = benchmark_ys['test']

    benchmark_betas = np.linalg.pinv(add_constant(benchmark_X_train)) @ benchmark_y_train
    benchmark_preds = benchmark_betas @ add_constant(benchmark_X_test).T

    benchmark_errs = dict()
    for m in samples.keys():  # choose a subset cardinality
        err_avgs = []
        for combo in samples[m]:
            if m == 1:
                combo = [combo]
            avg_pred = benchmark_preds[combo].mean()  # avg predicted outcome
            avg_target = benchmark_y_test[combo].mean()  # avg true outcome
            err_avgs.append((avg_pred - avg_target) ** 2)  # squared diff in avg pred

        benchmark_errs[m] = np.sqrt(np.stack(err_avgs).mean(axis=0))
            
    return grouped, benchmark_errs, beta_norm_dict
    