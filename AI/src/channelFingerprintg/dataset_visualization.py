import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from DatasetHandler import ChannelSpectrogram


def plot_data_scenario(data, index_start, filename, show_plot=True):
    example_data = {
        "AB": data[index_start],
        "AE": data[index_start + 1],
        "BA": data[index_start + 2],
        "BE": data[index_start + 3],
    }

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))

    for i, (label, sample) in enumerate(example_data.items()):
        axes[i, 0].plot(sample.real, label="Real")
        axes[i, 0].plot(sample.imag, label="Imaginary")
        axes[i, 0].set_title("IQ Samples " + label)
        axes[i, 0].set_xlabel("Sample Number")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].legend()

    # Adapt STFT settings to the probe length.
    # WiFi chan_est probes are typically 2x64 IQ samples (total 128), so
    # use FFT=64 to better match one channel-estimation set.
    sample_len = int(np.asarray(next(iter(example_data.values()))).shape[0])
    if sample_len <= 128:
        fft_len = 64 if sample_len >= 64 else max(2, sample_len)
    else:
        fft_len = min(2048, sample_len)
    overlap = min(fft_len // 2, max(0, fft_len - 1))
    channel_spectrogram = ChannelSpectrogram()
    for i, (label, sample) in enumerate(example_data.items()):
        spectrogram = channel_spectrogram._gen_single_channel_spectrogram(sample, fft_len, overlap)
        arr = np.asarray(spectrogram)
        axes[i, 1].imshow(arr, aspect="auto", cmap="jet")
        axes[i, 1].set_title("Spectrogram " + label)
        axes[i, 1].set_xlabel("Frequency")
        axes[i, 1].set_ylabel("Time")

    plt.tight_layout()
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close(fig)
    print("Saved plot to: ", filename)


# Plot the avg power of the data for each scenario
def plot_avg_power_scenario(data, scenario_start_index, data_per_scenario, filename, show_plot=True):
    avg_power = {"AB": [], "AE": [], "BA": [], "BE": []}
    i = 0
    data = data[scenario_start_index:scenario_start_index + data_per_scenario]
    while i < data.shape[0]:
        avg_power["AB"].append(np.mean(np.abs(data[i]) ** 2))
        avg_power["AE"].append(np.mean(np.abs(data[i + 1]) ** 2))
        avg_power["BA"].append(np.mean(np.abs(data[i + 2]) ** 2))
        avg_power["BE"].append(np.mean(np.abs(data[i + 3]) ** 2))
        i += 4

    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    for i, (label, power) in enumerate(avg_power.items()):
        axes[i].plot(power, label=label)
        axes[i].set_title("Avg Power of " + label)
        axes[i].set_xlabel("Sample Number")
        axes[i].set_ylabel("Avg Power (dB)")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(filename + ".png")
    if show_plot:
        plt.show()
    plt.close(fig)
    print("Saved plot to: ", filename)


def plot_avg_min_max_power_scenario(data, scenario_start_index, data_per_scenario, filename, show_plot=True):
    avg_power = {"AB": [], "AE": [], "BA": [], "BE": []}
    i = 0
    data = data[scenario_start_index:scenario_start_index + data_per_scenario]
    while i < data.shape[0]:
        # Get the average from the greatest and lowest values of each sample.
        avg_power["AB"].append(np.mean(np.abs([np.max(data[i]) ** 2, np.min(data[i]) ** 2])))
        avg_power["AE"].append(np.mean(np.abs([np.max(data[i + 1]) ** 2, np.min(data[i + 1]) ** 2])))
        avg_power["BA"].append(np.mean(np.abs([np.max(data[i + 2]) ** 2, np.min(data[i + 2]) ** 2])))
        avg_power["BE"].append(np.mean(np.abs([np.max(data[i + 3]) ** 2, np.min(data[i + 3]) ** 2])))
        i += 4

    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    for i, (label, power) in enumerate(avg_power.items()):
        axes[i].plot(power, label=label)
        axes[i].set_title("Avg Min Max Power of " + label)
        axes[i].set_xlabel("Sample Number")
        axes[i].set_ylabel("Avg Min Max Power (dB)")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(filename + ".png")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_dataset_scenarios(
    data,
    output_directory,
    file_prefix="TEST",
    data_per_scenario=400,
    show_plot=True,
):
    os.makedirs(output_directory, exist_ok=True)
    number_of_scenarios = data.shape[0] // data_per_scenario
    for i in range(number_of_scenarios):
        scenario_start_index = int(i * data_per_scenario)
        plot_data_scenario(
            data,
            scenario_start_index,
            os.path.join(output_directory, f"{file_prefix}_IQ_Spectrogram_scenario_{i}.png"),
            show_plot=show_plot,
        )
        plot_avg_min_max_power_scenario(
            data,
            scenario_start_index,
            data_per_scenario,
            os.path.join(output_directory, f"{file_prefix}_Avg_Min_Max_Power_scenario_{i}"),
            show_plot=show_plot,
        )


def _normalize_data_type_name(x):
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl == "iq":
            return "IQ"
        if xl == "polar":
            return "Polar"
        if xl in ("spectrogram", "spectogram"):
            return "Spectrogram"
    return x


def plot_bdr_results_CSV(
    csv_file,
    title,
    save_path,
    EveRayTracing=False,
    img_type=".png",
    title_font_size=25,
    label_font_size=25,
    legend_font_size=16,
    tick_font_size=13,
):
    """Create grouped bar charts of BDR vs key size L for each input type."""
    from matplotlib.patches import Patch

    results = pd.read_csv(csv_file)
    if "data_type" not in results.columns:
        raise ValueError("CSV missing required column 'data_type'.")
    results["data_type_norm"] = results["data_type"].apply(_normalize_data_type_name)

    if "L" not in results.columns or results["L"].dropna().empty:
        if "Models" in results.columns:
            extracted = results["Models"].str.extract(r"_out(\d+)")[0]
            results["L"] = pd.to_numeric(extracted, errors="coerce")
        else:
            raise ValueError("CSV missing 'L' and 'Models' to derive L.")
    else:
        results["L"] = pd.to_numeric(results["L"], errors="coerce")
    results = results.dropna(subset=["L"]).copy()
    results["L"] = results["L"].astype(int)

    required_cols = {"quantization_method", "KDR_AB", "KDR_AC", "KDR_BC"}
    missing = required_cols.difference(results.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    metric_names = ["KDR_AB", "KDR_AC", "KDR_BC"]
    metric_labels = {"KDR_AB": "Alice-Bob", "KDR_AC": "Alice-Eve", "KDR_BC": "Bob-Eve"}
    metric_colors = {"KDR_AB": "#d62728", "KDR_AC": "#1f77b4", "KDR_BC": "#2ca02c"}
    if EveRayTracing:
        metric_names.append("KDR_AA")
        metric_names.append("KDR_BB")
        metric_labels["KDR_AA"] = "Alice-Eve (Ray Tracing)"
        metric_labels["KDR_BB"] = "Bob-Eve (Ray Tracing)"
        metric_colors["KDR_AA"] = "#ff7f0e"
        metric_colors["KDR_BB"] = "#800080"
    hatch_styles = ["", "//", "x", "-", ".", "o", "*", "+", "O"]

    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = img_type

    desired_types = ["IQ", "Polar", "Spectrogram"]
    present_types = [t for t in desired_types if t in results["data_type_norm"].unique().tolist()]
    print("Present types: ", present_types)
    for dtype in present_types:
        sub = results[results["data_type_norm"] == dtype].copy()
        if sub.empty:
            continue

        L_values = sorted(sub["L"].unique().tolist())
        quant_methods = list(dict.fromkeys(sub["quantization_method"].astype(str).tolist()))
        num_L = len(L_values)
        num_metrics = len(metric_names)
        num_quants = max(1, len(quant_methods))

        grouped = sub.groupby(["L", "quantization_method"], as_index=False)[metric_names].mean()
        hatch_map = {qm: hatch_styles[i % len(hatch_styles)] for i, qm in enumerate(quant_methods)}
        x_base = np.arange(num_L)
        group_width = 0.8
        metric_slot_width = group_width / (num_metrics - 2) if EveRayTracing else num_metrics
        bar_width = metric_slot_width / num_quants

        fig, ax = plt.subplots(figsize=(10, 6))
        for m_idx, metric in enumerate(metric_names):
            for q_idx, qname in enumerate(quant_methods):
                x_offsets = -group_width / 2 + (
                    ((m_idx - 2) if EveRayTracing and metric in ["KDR_AA", "KDR_BB"] else m_idx) * metric_slot_width
                ) + q_idx * bar_width + bar_width / 2
                xs = x_base + x_offsets
                heights = []
                for Lval in L_values:
                    row = grouped[(grouped["L"] == Lval) & (grouped["quantization_method"].astype(str) == qname)]
                    heights.append(float(row.iloc[0][metric]) if not row.empty else 0.0)
                ax.bar(
                    xs,
                    heights,
                    width=bar_width,
                    color=metric_colors[metric],
                    hatch=hatch_map[qname],
                    edgecolor="black",
                    label=None,
                    alpha=1,
                )
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
        ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(Lv) for Lv in L_values], fontsize=tick_font_size)
        ax.yaxis.set_tick_params(labelsize=tick_font_size)
        ax.set_xlabel("Binary Feature Vector Length (B)", fontsize=label_font_size)
        ax.set_ylabel(r"$\overline{\mathrm{BDR}}$", fontsize=label_font_size)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        color_handles = [
            Patch(facecolor=metric_colors[m], edgecolor="black", label=metric_labels[m])
            for m in metric_names
        ]
        legend1 = ax.legend(handles=color_handles, loc="upper left", fontsize=legend_font_size)
        ax.add_artist(legend1)

        if title:
            ax.set_title(title, fontsize=title_font_size)
        plt.tight_layout()
        out_path = f"{base}_{dtype}{ext}"
        print("Saving BDR plot to: ", out_path)
        plt.savefig(out_path)
        plt.close(fig)


def plot_bdr_results_CSV_for_scenario(
    csv_files,
    title,
    save_path,
    EveRayTracing=False,
    img_type=".png",
    title_font_size=25,
    label_font_size=25,
    legend_font_size=16,
    tick_font_size=13,
):
    from matplotlib.patches import Patch

    fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    Titles = ["Indoor Scenario", "Outdoor Scenario 1", "Outdoor Scenario 2"]
    for i, csv_file in enumerate(csv_files):
        results = pd.read_csv(csv_file)

        if "data_type" not in results.columns:
            raise ValueError("CSV missing required column 'data_type'.")
        results["data_type_norm"] = results["data_type"].apply(_normalize_data_type_name)

        if "L" not in results.columns or results["L"].dropna().empty:
            if "Models" in results.columns:
                extracted = results["Models"].str.extract(r"_out(\d+)")[0]
                results["L"] = pd.to_numeric(extracted, errors="coerce")
            else:
                raise ValueError("CSV missing 'L' and 'Models' to derive L.")
        else:
            results["L"] = pd.to_numeric(results["L"], errors="coerce")
        results = results.dropna(subset=["L"]).copy()
        results["L"] = results["L"].astype(int)

        required_cols = {"quantization_method", "KDR_AB", "KDR_AC", "KDR_BC"}
        missing = required_cols.difference(results.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        metric_names = ["KDR_AB", "KDR_AC", "KDR_BC"]
        metric_labels = {"KDR_AB": "Alice-Bob", "KDR_AC": "Alice-Eve", "KDR_BC": "Bob-Eve"}
        metric_colors = {"KDR_AB": "#d62728", "KDR_AC": "#1f77b4", "KDR_BC": "#2ca02c"}
        hatch_styles = ["", "//", "x", "-", ".", "o", "*", "+", "O"]
        if EveRayTracing and i > 0:
            metric_names.append("KDR_AA")
            metric_names.append("KDR_BB")
            metric_labels["KDR_AA"] = "Alice-Eve (Ray Tracing)"
            metric_labels["KDR_BB"] = "Bob-Eve (Ray Tracing)"
            metric_colors["KDR_AA"] = "#ff7f0e"
            metric_colors["KDR_BB"] = "#800080"

        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = img_type

        desired_types = ["IQ", "Polar", "Spectrogram"]
        present_types = [t for t in desired_types if t in results["data_type_norm"].unique().tolist()]
        print("Present types: ", present_types)
        if not present_types:
            continue
        dtype = "Spectrogram" if "Spectrogram" in present_types else present_types[0]

        sub = results[results["data_type_norm"] == dtype].copy()
        if sub.empty:
            continue

        L_values = sorted(sub["L"].unique().tolist())
        quant_methods = list(dict.fromkeys(sub["quantization_method"].astype(str).tolist()))
        num_L = len(L_values)
        num_metrics = len(metric_names)
        num_quants = max(1, len(quant_methods))
        grouped = sub.groupby(["L", "quantization_method"], as_index=False)[metric_names].mean()
        hatch_map = {qm: hatch_styles[idx % len(hatch_styles)] for idx, qm in enumerate(quant_methods)}
        x_base = np.arange(num_L)
        group_width = 0.8
        metric_slot_width = group_width / max(1, (num_metrics - 2)) if (EveRayTracing and i > 0) else group_width / max(1, num_metrics)
        bar_width = metric_slot_width / num_quants

        ax = axs[i]
        ax.set_title(Titles[i], fontsize=title_font_size, loc="right")
        for m_idx, metric in enumerate(metric_names):
            for q_idx, qname in enumerate(quant_methods):
                x_offsets = (
                    -group_width / 2
                    + ((m_idx - 2) if (EveRayTracing and i > 0 and metric in ["KDR_AA", "KDR_BB"]) else m_idx) * metric_slot_width
                    + q_idx * bar_width
                    + bar_width / 2
                )
                xs = x_base + x_offsets
                heights = []
                for Lval in L_values:
                    row = grouped[(grouped["L"] == Lval) & (grouped["quantization_method"].astype(str) == qname)]
                    heights.append(float(row.iloc[0][metric]) if not row.empty else 0.0)
                ax.bar(xs, heights, width=bar_width, color=metric_colors[metric], hatch=hatch_map[qname], edgecolor="black", label=None, alpha=1)
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
        ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(Lv) for Lv in L_values], fontsize=tick_font_size)
        ax.yaxis.set_tick_params(labelsize=tick_font_size)
        if i == len(csv_files) - 1:
            ax.set_xlabel("Binary Feature Vector Length (B)", fontsize=label_font_size)
            ax.xaxis.set_tick_params(labelsize=label_font_size * 0.9)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(r"$\overline{\mathrm{BDR}}$", fontsize=label_font_size)
        ax.set_ylim(0, 0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        color_handles = [
            Patch(facecolor=metric_colors[m], edgecolor="black", label=metric_labels[m])
            for m in metric_names
        ]
        legend1 = ax.legend(handles=color_handles, loc="upper left", fontsize=legend_font_size)
        ax.add_artist(legend1)

    if title:
        fig.suptitle(title, fontsize=title_font_size)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)
    print("Saved figure to: ", save_path)


def plot_rec_rate_by_S(csv_path, title=None, save_path=None):
    """Plot rec_rate_AB vs S for each unique (L,bps,K) line from a results CSV."""
    from matplotlib import cm
    from matplotlib.lines import Line2D

    df = pd.read_csv(csv_path)
    required_cols = {"S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    if "data_type" in df.columns:
        mask = df["data_type"].astype(str).str.strip().str.lower().isin(["spectrogram", "spectogram"])
        df = df[mask].copy()
        if df.empty:
            raise ValueError("No rows with data_type equal to 'spectrogram'/'spectogram' found in the CSV.")

    df["S"] = pd.to_numeric(df["S"], errors="coerce")
    df["rec_rate_AB"] = pd.to_numeric(df["rec_rate_AB"], errors="coerce")
    df["rec_rate_AC"] = pd.to_numeric(df["rec_rate_AC"], errors="coerce")
    df["rec_rate_BC"] = pd.to_numeric(df["rec_rate_BC"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["bps"] = pd.to_numeric(df["bps"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"]).copy()

    agg = (
        df.groupby(["L", "bps", "K", "S"], as_index=False)[["rec_rate_AB", "rec_rate_AC", "rec_rate_BC"]].mean()
        .sort_values(["L", "bps", "K", "S"])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = agg[["L", "bps", "K"]].drop_duplicates().reset_index(drop=True)
    cmap = cm.get_cmap("tab20", len(groups))
    group_to_color = {tuple(groups.loc[i, ["L", "bps", "K"]].astype(int)): cmap(i) for i in range(len(groups))}
    pair_to_col = {
        "Alice-Bob": "rec_rate_AB",
        "Alice-Eve": "rec_rate_AC",
        "Bob-Eve": "rec_rate_BC",
    }
    pair_to_style = {"Alice-Bob": "-", "Alice-Eve": "--", "Bob-Eve": ":"}

    for (L_val, bps_val, K_val), sub in agg.groupby(["L", "bps", "K"], sort=False):
        color = group_to_color[(int(L_val), int(bps_val), int(K_val))]
        for pair, col_name in pair_to_col.items():
            label = f"L={int(L_val)}, bps={int(bps_val)}, K={int(K_val)}" if pair == "Alice-Bob" else None
            ax.plot(
                sub["S"].values,
                sub[col_name].values,
                marker="o" if pair == "Alice-Bob" else None,
                linestyle=pair_to_style[pair],
                color=color,
                label=label,
            )

    ax.set_xlabel("Parity Symbol Length S")
    ax.set_ylabel("Reconciliation Rate")
    ax.set_title(title if title is not None else "Reconciliation Rate for Parity Symbol Length S")
    ax.grid(True, linestyle="--", alpha=0.4)
    group_legend = ax.legend(loc="best", title="Groups (L,bps,K)")
    ax.add_artist(group_legend)
    style_handles = [
        Line2D([0], [0], color="black", linestyle=pair_to_style[pair], label=pair)
        for pair in ["Alice-Bob", "Alice-Eve", "Bob-Eve"]
    ]
    ax.legend(handles=style_handles, title="Node Pair", loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_rec_rate_by_S_for_L(
    csv_path,
    title=None,
    save_path=None,
    L=512,
    quantization_method="threshold",
    title_font_size=25,
    label_font_size=25,
    legend_font_size=16,
    tick_font_size=13,
):
    """Plot rec_rate vs RS code rate for a single key size L from a results CSV."""
    from matplotlib import cm
    from matplotlib.lines import Line2D

    df = pd.read_csv(csv_path)
    required_cols = {"S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    if "data_type" in df.columns:
        mask = df["data_type"].astype(str).str.strip().str.lower().isin(["spectrogram", "spectogram"])
        df = df[mask].copy()
        if df.empty:
            raise ValueError("No rows with data_type equal to 'spectrogram'/'spectogram' found in the CSV.")

    df["S"] = pd.to_numeric(df["S"], errors="coerce")
    df["rec_rate_AB"] = pd.to_numeric(df["rec_rate_AB"], errors="coerce")
    df["rec_rate_AC"] = pd.to_numeric(df["rec_rate_AC"], errors="coerce")
    df["rec_rate_BC"] = pd.to_numeric(df["rec_rate_BC"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["bps"] = pd.to_numeric(df["bps"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"]).copy()

    target_L = int(L)
    df = df[df["L"] == target_L].copy()
    if df.empty:
        raise ValueError(f"No rows found for L={target_L} after filtering.")

    df = df[df["quantization_method"] == quantization_method].copy()
    agg = (
        df.groupby(["bps", "K", "S"], as_index=False)[["rec_rate_AB", "rec_rate_AC", "rec_rate_BC"]].mean()
        .sort_values(["bps", "K", "S"])
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    groups = agg[["bps", "K"]].drop_duplicates().reset_index(drop=True)
    cmap = cm.get_cmap("tab20", len(groups))
    group_to_color = {tuple(groups.loc[i, ["bps", "K"]].astype(int)): cmap(i) for i in range(len(groups))}
    pair_to_col = {
        "Alice-Bob": "rec_rate_AB",
        "Alice-Eve": "rec_rate_AC",
        "Bob-Eve": "rec_rate_BC",
    }
    pair_to_style = {"Alice-Bob": "-", "Alice-Eve": "--", "Bob-Eve": ":"}
    for (bps_val, K_val), sub in agg.groupby(["bps", "K"], sort=False):
        color = group_to_color[(int(bps_val), int(K_val))]
        for pair, col_name in pair_to_col.items():
            label = f"Z={int(bps_val)}" if pair == "Alice-Bob" else None
            ax.plot(
                (K_val) / (K_val + sub["S"].values),
                sub[col_name].values,
                marker="o" if pair == "Alice-Bob" else None,
                linestyle=pair_to_style[pair],
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )
            if pair == "Alice-Bob":
                x_vals = (K_val) / (K_val + sub["S"].values)
                y_vals = sub[col_name].values
                n_vals = (K_val + sub["S"].values).astype(int)
                for xv, yv, nv in zip(x_vals, y_vals, n_vals):
                    ax.annotate(
                        f"RS({int(nv)},{int(K_val)})",
                        xy=(xv, yv),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=legend_font_size,
                        color=color,
                        zorder=5,
                    )

    ax.set_xlabel("RS Code Rate K/N")
    ax.xaxis.label.set_fontsize(label_font_size)
    ax.set_ylabel("Average Reconciliation Rate")
    ax.yaxis.label.set_fontsize(label_font_size)
    plot_title = title if title is not None else f"Average Reconciliation Rate vs RS Code Rate (B={target_L})"
    ax.set_title(plot_title)
    ax.title.set_fontsize(title_font_size)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
    ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
    ax.tick_params(axis="x", labelsize=tick_font_size)
    ax.tick_params(axis="y", labelsize=tick_font_size)

    group_legend = ax.legend(
        loc="upper left",
        title="Bits per Symbol",
        fontsize=legend_font_size,
        bbox_to_anchor=(0, 0.78),
    )
    ax.add_artist(group_legend)
    style_handles = [
        Line2D([0], [0], color="black", linestyle=pair_to_style[pair], label=pair)
        for pair in ["Alice-Bob", "Alice-Eve", "Bob-Eve"]
    ]
    ax.legend(handles=style_handles, title="Node Pair", loc="upper left", fontsize=legend_font_size)
    ax.invert_xaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
