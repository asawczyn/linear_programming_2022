from matplotlib import pyplot as plt


def plot_gant(df):
    MAX_TIME = 12
    fig, gnt = plt.subplots()
    gnt.set_ylim(8, 41)
    gnt.set_xlim(0, MAX_TIME)

    gnt.set_xlabel("Starting time")
    gnt.set_ylabel("Machine")

    gnt.set_yticks([15, 25, 35])
    gnt.hlines(
        y=[9.5, 19.5, 29.5, 39.5],
        xmin=[0],
        xmax=[MAX_TIME],
        colors="purple",
        linestyles="--",
        lw=1,
        label="Multiple Lines",
    )

    gnt.set_yticklabels(["3", "2", "1"])
    gnt.set_xticks(list(range(MAX_TIME + 1)))

    for i, color in zip(range(1, df.job.max() + 1), ["tab:orange", "tab:red", "tab:blue"]):
        for ij_, s_, p_, m_ in df[df.job == i][
            ["i,j", "starting_time", "processing_time", "machine"]
        ].values:
            m_ = (df.machine.max() - m_ + 1) * 10
            gnt.broken_barh([(s_, p_)], (m_ + (i - 1) * 3, 3), facecolors=color)
            gnt.text(
                x=s_ + p_ / 2,
                y=m_ + (i - 1) * 3 + 1.2,
                s=f"O_{ij_[0]},{ij_[1]}",
                ha="center",
                va="center",
                size=8,
            )

    plt.savefig("job_shop_gant.png")
    plt.show()