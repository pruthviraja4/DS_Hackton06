import matplotlib.pyplot as plt
import pandas as pd
def quick_hist(df, col, out=None):
    plt.figure(figsize=(6,4))
    df[col].hist(bins=50)
    plt.title(col)
    if out:
        plt.savefig(out)
    else:
        plt.show()
