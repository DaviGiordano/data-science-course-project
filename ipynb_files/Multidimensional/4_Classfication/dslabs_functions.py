'''
file:       dslabs_functions.py
version:    2023.1
'''
from math import pi, sin, cos
from itertools import product
from datetime import datetime
from numbers import Number
from numpy import arange, ndarray, set_printoptions
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes
from matplotlib.pyplot import gca, gcf, savefig, subplots
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
#from matplotlib.dates import _reset_epoch_test_example, set_epoch
from pandas import DataFrame, read_csv, concat, unique, to_numeric, to_datetime
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from config import ACTIVE_COLORS, LINE_COLOR, FILL_COLOR, cmap_blues

NR_COLUMNS: int = 3
HEIGHT: int = 4

TEXT_MARGIN = 0.05
FONT_SIZE = 6
FONT_TEXT = FontProperties(size=FONT_SIZE)

alpha = 0.3

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

#_reset_epoch_test_example()
#set_epoch('0000-12-31T00:00:00')  # old epoch (pre MPL 3.3)

# ---------------------------------------
#             DATA CHARTS 
# ---------------------------------------

def define_grid(nr_vars, vars_per_row: int= NR_COLUMNS) -> tuple:
    nr_rows = 1
    if nr_vars % vars_per_row == 0:
        nr_rows = nr_vars // vars_per_row
    else:
        nr_rows = nr_vars // vars_per_row + 1
    return nr_rows, vars_per_row

def set_chart_labels(ax, title: str='', xlabel: str='', ylabel: str=''):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def set_chart_xticks(xvalues: list, ax: Axes, percentage: bool=False):
    if len(xvalues) > 0:    
        if percentage:
            ax.set_ylim(0.0, 1.0)
        
        if isinstance(xvalues[0], datetime):
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
        
        rotation = 0
        if isinstance(xvalues[0], Number):
            ax.set_xlim((xvalues[0], xvalues[-1]))
            ax.set_xticks(xvalues, labels=xvalues)
        else:
            rotation = 45

        ax.tick_params(axis='x', labelrotation=rotation, labelsize='xx-small')
        
    return ax

def plot_line_chart(xvalues: list, yvalues: list, ax: Axes=None, title: str='', xlabel: str='', 
                           ylabel: str='', percentage: bool=False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax, percentage=percentage)
    ax.plot(xvalues, yvalues, c=LINE_COLOR)
    return ax

def plot_bar_chart(xvalues: list, yvalues: list, ax: Axes=None, title: str='', xlabel: str='', 
                   ylabel: str='', percentage: bool=False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    values = ax.bar(xvalues, yvalues, label=yvalues, edgecolor=LINE_COLOR, color=FILL_COLOR, tick_label=xvalues)
    format = '%.2f' if percentage else '%.0f'
    ax.bar_label(values, fmt=format, fontproperties=FONT_TEXT)

    return ax

def plot_scatter_chart(var1: list, var2: list, ax: Axes=None, title: str='', xlabel: str='', ylabel: str=''):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.scatter(var1, var2)
    return ax

def plot_multiline_chart(xvalues: list, yvalues: dict, ax: Axes=None, title: str='', xlabel: str='',
                        ylabel: str='', percentage: bool=False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)
    return ax

def plot_and_save_multiline_chart(xvalues: list, yvalues: dict, ax: Axes = None, title: str = '', xlabel: str = '',
                         ylabel: str = '', percentage: bool = False, save_path: str = None):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)
    if save_path:
        savefig(save_path)
    return ax


def plot_multibar_chart(group_labels: list, yvalues: dict, ax: Axes=None, title: str='', 
                        xlabel: str='', ylabel: str='', percentage: bool=False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    bar_labels = list(yvalues.keys())

    # This is the location for each bar
    index = arange(len(group_labels))
    bar_width = 0.8 / len(bar_labels)
    ax.set_xticks(index + bar_width / 2, labels=group_labels)

    for i in range(len(bar_labels)):
        values = ax.bar(index + i * bar_width, yvalues[bar_labels[i]], width=bar_width, label=bar_labels[i])
        format = '%.2f' if percentage else '%.0f'
        ax.bar_label(values, fmt=format, fontproperties=FONT_TEXT)
    ax.legend(fontsize=FONT_SIZE)
    return ax

def plot_multi_scatters_chart(data: DataFrame, var1: str, var2: str, var3: str=None, ax: Axes=None):
    if ax is None:
        ax = gca()

    title=f'{var1} x {var2}'
    if var3 is not None:
        title += f' per {var3}'
        values = data[var3].unique().tolist()
        if len(values) > 2:
            chart = ax.scatter(data[var1], data[var2], c=data[var3])
            cbar = gcf().colorbar(chart)
            cbar.outline.set_visible(False)
            cbar.set_label(var3, loc='top')
        else:
            values.sort()
            for i in range(len(values)):
                subset = data[data[var3] == values[i]]
                ax.scatter(subset[var1], subset[var2], color=ACTIVE_COLORS[i], label=values[i])
            ax.legend(fontsize='xx-small')
    else:
        ax.scatter(data[var1], data[var2], color=FILL_COLOR)
    ax = set_chart_labels(ax=ax, title=title, xlabel=var1, ylabel=var2)
    return ax

def plot_horizontal_bar_chart(elements: list, values: list, error: list=None, ax: Axes=None, title: str = '', 
                              xlabel: str = '', ylabel: str = '', percentage: bool=False):
    if ax is None:
        ax = gca()
    if percentage:
        ax.set_xlim((0,1))
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    y_pos = arange(len(elements))

    ax.barh(y_pos, values, xerr=error, align='center', error_kw={'lw': 0.5, 'ecolor': 'r'})
    ax.set_yticks(y_pos, labels=elements)
    ax.invert_yaxis()  # labels read top-to-bottom
    return ax

# ---------------------------------------
#             DATA PROFILING 
# ---------------------------------------
def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {'numeric': [], 'binary': [], 'date': [], 'symbolic': []}

    nr_values = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types['binary'].append(c)
            df[c].astype('bool')
        else:
            try:
                to_numeric(df[c], errors='raise')
                variable_types['numeric'].append(c)
            except:
                try:
                    df[c] = to_datetime(df[c], errors='raise')
                    variable_types['date'].append(c)
                except:
                    variable_types['symbolic'].append(c)

    return variable_types

def determine_outlier_thresholds_for_var(summary5, std_based: bool=True, threshold: int=NR_STDEV):
    top, bottom = 0, 0
    if std_based:
        std = threshold * summary5['std']
        top = summary5['mean'] + std
        bottom = summary5['mean'] - std
    else:
        iqr = threshold * (summary5['75%'] - summary5['25%'])
        top = summary5['75%'] + iqr
        bottom = summary5['25%'] - iqr
            
    return top, bottom

def count_outliers(data: DataFrame, numeric, nrstdev: int = NR_STDEV, iqrfactor: float = IQR_FACTOR):
    outliers_iqr = []
    outliers_stdev = []
    summary5 = data[numeric].describe()

    for var in numeric:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=True, threshold=nrstdev)
        outliers_stdev += [data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]]
    
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=iqrfactor)
        outliers_iqr += [data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]]
        
    return {'iqr': outliers_iqr, 'stdev': outliers_stdev}

def analyse_date_granularity(data, var, levels, file_tag=''):
    cols = len(levels)
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f'Granularity study for {var}')

    for i in range(cols):
        counts = data[var+'_'+levels[i]].value_counts()
        plot_bar_chart(counts.index.to_list(), counts.values, ax=axs[0][i], title=levels[i], xlabel=levels[i], ylabel='nr records', percentage=False)
    savefig(f'images/{file_tag}_granularity_{var}.png')

# ---------------------------------------
#             DATA PREPARATION
# ---------------------------------------

def derive_date_variables(df, date_vars):
    for date in date_vars:
        df[date+'_year'] = df[date].dt.year
        df[date+'_quarter'] = df[date].dt.quarter
        df[date+'_month'] = df[date].dt.month
        df[date+'_day'] = df[date].dt.day
    return df 

def encode_cyclic_variables(data, vars):
    for v in vars:
        x_max = max(data[v])
        data[v+'_sin'] = data[v].apply(lambda x: round(sin(2*pi*x/x_max), 3))
        data[v+'_cos'] = data[v].apply(lambda x: round(cos(2*pi*x/x_max), 3))

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=bool, drop='if_binary')
    trans = enc.fit_transform(df[vars_to_dummify])
    print(trans)

    new_vars = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

def mvi_by_dropping(data: DataFrame, min_pct_per_variable:float=0.1, min_pct_per_record: float=0.0) -> DataFrame:
    '''
        data: DataFrame - the data to clean
        min_pct_per_variable: float - the minimum percentage of records a variable has to show in order to be kept
        min_pct_per_record: float - the minimum percentage of values that a record has to show in order to be kept
        return the data modified
    '''
    # Deleting variables
    df = data.dropna(axis=1, thresh=data.shape[0]*min_pct_per_variable, inplace=False)
    # Deleting records
    df.dropna(axis=0, thresh=data.shape[1]*min_pct_per_record, inplace=True)

    return df

def mvi_by_filling(data: DataFrame, strategy: str ='frequent') -> DataFrame:
    '''
        data: DataFrame - the data to clean
        strategy: str - the strategy to apply ('frequent', 'constant' or 'knn')
        return the data modified
    '''
    df, tmp_nr, tmp_sb, tmp_bool = None, None, None, None
    variables = get_variable_types(data)

    stg_num, v_num = 'mean', -1
    stg_sym, v_sym = 'most_frequent', 'NA'
    stg_bool, v_bool = 'most_frequent', False
    if strategy != 'knn':
        if strategy == 'constant':
            stg_num, stg_sym, stg_bool = 'constant', 'constant', 'constant'
        if len(variables['numeric']) > 0:
            imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
            tmp_nr = DataFrame(imp.fit_transform(data[variables['numeric']]), columns=variables['numeric'])
        if len(variables['symbolic']) > 0:
            imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
            tmp_sb = DataFrame(imp.fit_transform(data[variables['symbolic']]), columns=variables['symbolic'])
        if len(variables['binary']) > 0:
            imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
            tmp_bool = DataFrame(imp.fit_transform(data[variables['binary']]), columns=variables['binary'])

        df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    else:
        imp = KNNImputer(n_neighbors=5)
        imp.fit(data)
        ar = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df

# ---------------------------------------
#             CLASSIFICATION
# ---------------------------------------

DELTA_IMPROVE: float = 0.001

CLASS_EVAL_METRICS = {
    'accuracy': accuracy_score, 
    'recall': recall_score, 
    'precision': precision_score, 
    'auc': roc_auc_score, 
    'f1': f1_score,
}

def read_train_test_from_files(train_fn: str, test_fn: str, target: str='class'):
    train = read_csv(train_fn, index_col=None)
    trnY = train.pop(target).values
    trnX = train.values
    labels = unique(trnY)
    labels.sort()

    test  = read_csv(test_fn, index_col=None)
    tstY = test.pop(target).values
    tstX = test.values
    return trnX, tstX, trnY, tstY, labels, train.columns

def plot_confusion_matrix(cnf_matrix: ndarray, classes_names: ndarray, ax: Axes=None):
    if ax is None:
        ax = gca()
    title = 'Confusion matrix'
    set_printoptions(precision=2)
    tick_marks = arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap_blues)

    for i, j in product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(j, i, format(cnf_matrix[i, j], 'd'), color='y', horizontalalignment="center")
    return ax

def plot_roc_chart(tstY: ndarray, predictions: dict, ax: Axes=None, target: str = 'class'):
    if ax is None:
        ax = gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)

    ax.plot([0, 1], [0, 1], color='navy', label='random', linewidth=1, linestyle='--',  marker='')
    models = list(predictions.keys())
    for i in range(len(models)):
        RocCurveDisplay.from_predictions(y_true=tstY, y_pred=predictions[models[i]], name=models[i], 
                                         ax=ax, color=ACTIVE_COLORS[i], linewidth=1) 
    ax.legend(loc="lower right")
    return ax

def plot_evaluation_results(model, trn_y, prd_trn, tst_y, prd_tst, labels: ndarray, file_tag=''):
    evaluation = {}
    for key in CLASS_EVAL_METRICS:
        evaluation[key] = [CLASS_EVAL_METRICS[key](trn_y, prd_trn), CLASS_EVAL_METRICS[key](tst_y, prd_tst)]

    params_st = '' if () == model['params'] else str(model['params'])
    fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    fig.suptitle('Best {} for {} {}'.format(model['metric'], model['name'], params_st))
    plot_multibar_chart(['Train', 'Test'], evaluation, ax=axs[0], percentage=True)

    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])
    savefig('images/' + file_tag + '_' + model['name'] + '_best_' + model['metric'] + '_eval.png')
    return axs

def naive_Bayes_study(trnX, trnY, tstX, tstY, metric='accuracy', file_tag=''):
    estimators = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }

    xvalues = []
    yvalues = []
    best_model = None
    best_params = {'name': '', 'metric': metric, 'params': ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance = eval
            best_params['name'] = clf
            best_model = estimators[clf]
        yvalues.append(eval)

    plot_bar_chart(xvalues, yvalues, title=f'Naive Bayes Models ({metric})', ylabel=metric, percentage=True)
    savefig(f'images/{file_tag}_nb_{metric}_study.png')
    
    return best_model, best_params

def knn_study(trnX, trnY, tstX, tstY, k_max=19, lag=2, metric='accuracy', file_tag=''):
    dist = ['manhattan', 'euclidean', 'chebyshev']

    kvalues = [i for i in range(1, k_max+1, lag)]
    best_model = None
    best_params = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance = 0

    values = {}
    for d in dist:
        y_tst_values = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (k, d)
                best_model = clf
        values[d] = y_tst_values
    print('KNN best with k={} and {}'.format(best_params['params'][0], best_params['params'][1]))
    plot_multiple_line_chart(kvalues, values, title='KNN Models ({})'.format(metric), xlabel='k', ylabel=metric, percentage=True)
    savefig(f'images/{file_tag}_knn_{metric}_study.png')

    return best_model, best_params

