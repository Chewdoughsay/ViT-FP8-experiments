import torch
import json
import os
import sys
from pathlib import Path


def calculate_stats(epoch_times):
    """Calculeaza statistici medii si totale."""
    if not epoch_times:
        return 0, 0
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times)
    return total_time, avg_time


def inspect_checkpoint(checkpoint_path):
    """Incarca si extrage metricele cheie dintr-un fisier checkpoint."""
    try:
        # Incarca fisierul pe CPU pentru compatibilitate
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        return {"error": f"Eroare la incarcarea fisierului: {e}"}

    metrics = checkpoint.get('metrics', {})

    # Gaseste valoarea finala a metricilor
    train_loss_final = metrics.get('train_loss', [0])[-1]
    val_loss_final = metrics.get('val_loss', [0])[-1]
    train_acc_final = metrics.get('train_acc', [0])[-1]
    val_acc_final = metrics.get('val_acc', [0])[-1]

    # Extrage timpul si calculeaza media
    epoch_times = metrics.get('epoch_time', [])
    total_time, avg_time = calculate_stats(epoch_times)

    return {
        "Best Val Acc": checkpoint.get('val_acc', 0),  # Acuratetea maxima la care s-a salvat best_model
        "Final Val Acc": val_acc_final,
        "Final Train Loss": train_loss_final,
        "Final Val Loss": val_loss_final,
        "Avg Epoch Time (s)": avg_time,
        "Total Time (min)": total_time / 60,
        "Epochs": len(metrics.get('train_loss', []))
    }


def collect_experiment_metrics(experiment_name, base_dir='experiments/results/checkpoints'):
    """Colecteaza metricele finale din folderul unui experiment."""

    # Formeaza calea catre folderul experimentului
    exp_dir = Path(base_dir) / experiment_name

    # Calea catre cel mai bun model
    best_model_path = exp_dir / 'best_model.pt'

    print(f"\n🔬 Analizing Experiment: {experiment_name}")
    print("--------------------------------------------------")

    if not exp_dir.exists():
        return {"Experiment": experiment_name, "Status": "❌ Folderul nu exista"}

    if not best_model_path.exists():
        print(f"⚠️ Fisierul 'best_model.pt' nu a fost gasit in {exp_dir}. Incarc 'final_metrics.json'...")
        # Daca nu avem best_model.pt, incercam sa incarcam JSON-ul final
        metrics_json_path = exp_dir / 'final_metrics.json'

        if metrics_json_path.exists():
            try:
                with open(metrics_json_path, 'r') as f:
                    metrics = json.load(f)

                # Extrage direct din JSON
                val_acc = max(metrics.get('val_acc', [0]))
                epoch_times = metrics.get('epoch_time', [])
                total_time, avg_time = calculate_stats(epoch_times)

                # Reconstruim outputul pentru comparatie
                return {
                    "Experiment": experiment_name,
                    "Best Val Acc": val_acc,
                    "Final Train Acc": metrics.get('train_acc', [0])[-1],
                    "Final Val Loss": metrics.get('val_loss', [0])[-1],
                    "Avg Epoch Time (s)": avg_time,
                    "Total Time (min)": total_time / 60,
                    "Epochs": len(metrics.get('train_acc', []))
                }
            except Exception as e:
                return {"Experiment": experiment_name, "Status": f"❌ Eroare la incarcarea JSON: {e}"}
        else:
            return {"Experiment": experiment_name,
                    "Status": "❌ Nici best_model.pt, nici final_metrics.json nu au fost gasite."}

    # Daca am gasit best_model.pt
    results = inspect_checkpoint(best_model_path)
    results['Experiment'] = experiment_name

    # Extragem din nou metricele din JSON (daca exista) pentru a avea 'Final' Acc/Loss
    metrics_json_path = exp_dir / 'final_metrics.json'
    if metrics_json_path.exists():
        try:
            with open(metrics_json_path, 'r') as f:
                metrics = json.load(f)

            results["Final Val Acc"] = metrics.get('val_acc', [0])[-1]
            results["Final Train Acc"] = metrics.get('train_acc', [0])[-1]
            results["Final Val Loss"] = metrics.get('val_loss', [0])[-1]

            # Recalculăm timpul, deoarece checkpoint-ul nu salvează lista completă
            epoch_times = metrics.get('epoch_time', [])
            total_time, avg_time = calculate_stats(epoch_times)
            results["Avg Epoch Time (s)"] = avg_time
            results["Total Time (min)"] = total_time / 60
            results["Epochs"] = len(metrics.get('val_acc', []))

        except Exception:
            pass  # Ignoram eroarea, ne bazam pe datele din best_model

    return results


def print_summary(results_list):
    """Afiseaza rezultatele intr-un format tabelar curat."""
    if not results_list:
        print("Nu s-au gasit rezultate.")
        return

    # Determinare latimi maxime pentru coloane
    cols = ["Experiment", "Best Val Acc", "Final Train Acc", "Final Val Loss", "Avg Epoch Time (s)", "Total Time (min)"]

    # Formateaza numerele cu precizie pentru usurinta in comparatie
    data = []
    for r in results_list:
        if 'Status' in r:
            data.append([r.get("Experiment", "N/A"), r["Status"], "", "", "", ""])
            continue

        data.append([
            r['Experiment'],
            f"{r['Best Val Acc']:.4f} ({r['Best Val Acc'] * 100:.2f}%)",
            f"{r['Final Train Acc']:.4f} ({r['Final Train Acc'] * 100:.2f}%)",
            f"{r['Final Val Loss']:.4f}",
            f"{r['Avg Epoch Time (s)']:.2f}",
            f"{r['Total Time (min)']:.1f}"
        ])

    col_widths = [len(header) for header in cols]
    for row in data:
        for i, item in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(item)))

    # Afisare Header
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(cols))
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("=" * len(header_line))

    # Afisare Date
    for row in data:
        row_line = " | ".join(
            row[i].ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i]) for i in range(len(cols)))
        print(row_line)
    print("=" * len(header_line) + "\n")


if __name__ == "__main__":
    # LISTA EXPERIMENTELOR PENTRU COMPARATIE
    # Asigura-te ca denumirile folderelor se potrivesc cu ce ai tu in results/checkpoints/

    experiments_to_analyze = [
        # FP32 - Baseline (Exp 1)
        "baseline_fp32",
        # FP32 - Regularizat (Exp 2 - Cel mai bun model)
        "experiment2_regularized",
        # FP16 - Mixed Precision (Exp 3 - Cel mai nou)
        "experiment3_fp16"
    ]

    all_results = []
    for exp_name in experiments_to_analyze:
        result = collect_experiment_metrics(exp_name)
        # Adauga in lista doar daca nu a fost o eroare critica de fisier lipsa
        if not result.get("Status", "").startswith("❌"):
            all_results.append(result)
        else:
            print(result["Status"])

    print_summary(all_results)
    print("Date extrase si agregate. Acum poti compara acuratetea si viteza!")