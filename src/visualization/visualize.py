# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_pickle(r"../../data/interim/01_data_processed.pkl")

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"].reset_index(drop=True))


# plot all exercises


labels = df["label"].unique()
labels
for label in labels:
    subsets = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subsets[:100]["acc_y"].reset_index(drop=True))
    ax.legend()
    ax.set_title(f"Exercise: {label}")
    ax.set_xlabel("samples")
    ax.set_ylabel("acc_y")
    plt.show()


# adjust plot settings

mpl.rcParams["figure.figsize"] = (30, 10)
mpl.rcParams["figure.dpi"] = 100


# comparing medium vs heavy sets

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()

# compare participants
participant_df = (
    df.query("label == 'bench'").sort_values(by="participant").reset_index()
)


fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()


label = "squat"
participant = "A"

all_axis_df = df.query(f"label == '{label}'").query("participant == 'A'").reset_index()


fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("axis")
plt.legend()
plt.show()


labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
              .query(f"participant == '{participant}'")
              .reset_index()
        )
        if len(all_axis_df) == 0:
            continue
        fig, ax = plt.subplots()
        all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
        ax.set_xlabel("samples")
        ax.set_ylabel("axis")
        plt.legend()
        plt.title(f"{label} {participant}".title())
        plt.show()
        
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
              .query(f"participant == '{participant}'")
              .reset_index()
        )
        if len(all_axis_df) == 0:
            continue
        fig, ax = plt.subplots()
        all_axis_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
        ax.set_xlabel("samples")
        ax.set_ylabel("axis")
        plt.legend()
        plt.title(f"{label} {participant}".title())
        plt.show()
		
for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
			.query(f"participant == '{participant}'")
            .reset_index()
		)        
        if len(combined_plot_df) == 0:
            continue
        fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(20,10))
        combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
        combined_plot_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])
        ax[0].legend(
            loc = 'upper center',
            bbox_to_anchor = (0.5,1.15),
            ncols = 3,
            fancybox = True,
            shadow = True
		)
        ax[1].legend(
            loc = 'upper center',
            bbox_to_anchor = (0.5,1.15),
            ncols = 3,
            fancybox = True,
            shadow = True
		)
        ax[1].set_xlabel("samples")
        # reports -> figures
        plt.savefig(f'../../reports/figures/{label.title()} ({participant.title()}).png')
        plt.show()


		
