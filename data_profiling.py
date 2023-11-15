from dslabs_functions import set_chart_labels
from dslabs_functions import plot_multibar_chart
from pandas import Series
from pandas import read_csv, DataFrame, Series, to_numeric, to_datetime
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart
from matplotlib.pyplot import savefig, show
from dslabs_functions import get_variable_types
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, show, subplots
from dslabs_functions import define_grid, HEIGHT
from dslabs_functions import plot_bar_chart

covidData = read_csv("data/class_pos_covid.csv", na_values="")


def plot_records_vs_variables(data: DataFrame, file_tag: str):
    print("Plotting nr. records vs nr. variables for", file_tag)
    figure(figsize=(7, 5))
    values: dict[str, int] = {
        "nr records": data.shape[0], "nr variables": data.shape[1]}
    plot_bar_chart(
        list(values.keys()),
        list(values.values()),
        title="Nr of records vs nr variables"
    )
    savefig(f"images/{file_tag}_records_variables.png")


def plot_missing_data(data: DataFrame, file_tag: str):
    print("Plotting missing data for", file_tag)
    mv: dict[str, int] = {}
    for var in data.columns:
        nr: int = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure()
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values",
    )
    savefig(f"images/{file_tag}_mv.png")


def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [],
        "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types


def plot_variable_types(data: DataFrame, file_tag: str):
    print("Plotting variable types", file_tag)
    variable_types: dict[str, list] = get_variable_types(data)
    print(variable_types)
    counts: dict[str, int] = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])

    figure(figsize=(6, 4))
    plot_bar_chart(
        list(counts.keys()), list(counts.values()), title="Nr of variables per type"
    )
    savefig(f"images/{file_tag}_variable_types.png")


# symbolic: list[str] = variable_types["symbolic"]
# data[symbolic] = data[symbolic].apply(lambda x: x.astype("category"))
# data.dtypes


def plot_box_plots(data: DataFrame, file_tag: str):
    print("Plotting box plot for", file_tag)
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    if [] != numeric:
        data[numeric].boxplot(rot=45)
        savefig(f"images/{file_tag}_global_boxplot.png")
        show()
    else:
        print("There are no numeric variables.")

    if [] != numeric:
        rows: int
        cols: int
        rows, cols = define_grid(len(numeric))
        axs: ndarray
        _, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        i, j = 0, 0
        for n in range(len(numeric)):
            axs[i, j].set_title("Boxplot for %s" % numeric[n])
            axs[i, j].boxplot(data[numeric[n]].dropna().values)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/{file_tag}_single_boxplots.png")
        show()
    else:
        print("There are no numeric variables.")


NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

summary5: DataFrame = data.describe(include="all")

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom


def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top: float
        bottom: float
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var]
                                        < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var]
                                        < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


if [] != numeric:
    outliers: dict[str, int] = count_outliers(data, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"images/{file_tag}_outliers_standard.png")
    show()
else:
    print("There are no numeric variables.")


# -----------------------Histograms---------


if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i: int
    j: int
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data[numeric[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_single_histograms_numeric.png")
    show()
else:
    print("There are no numeric variables.")

# histogram symbolic


symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_histograms_symbolic.png")
    show()
else:
    print("There are no symbolic variables.")

# Class distribution

target = "CovidPos"

values: Series = data[target].value_counts()
print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"images/{file_tag}_class_distribution.png")
show()

'''
import matplotlib
#matplotlib.use('Agg')
from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart

data = data.dropna()

vars: list = data.columns.to_list()
'''
if [] != vars:
    target = "CovidPos"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
    savefig(f"images/{file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")

if [] != vars:
    target = "CovidPos"

    n: int = len(vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(
                data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/{file_tag}_sparsity_per_class_study.png")
    show()
else:
    print("Sparsity per class: there are no variables.")

'''


from seaborn import heatmap
from dslabs_functions import get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = data[numeric].corr().abs()

figure()
heatmap(
    abs(corr_mtx),
    xticklabels=numeric,
    yticklabels=numeric,
    annot=True,
    fmt=".2f", 
    cmap="Blues",
    vmin=0,
    vmax=1,
)
show()
savefig(f"images/{file_tag}_correlation_analysis.png")
