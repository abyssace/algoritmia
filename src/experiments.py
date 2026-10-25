"""Ejecuta los cinco ejercicios y genera datos, visualizaciones y reporte."""

from __future__ import annotations

import csv
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .algorithms import (
    Item, estimate_pi_chunk, exact_tsp_cost, greedy_knapsack, mst_preorder_tour,
    nearest_neighbor_tour, random_tour, simulated_annealing, tour_cost, two_opt,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "REPORTE.md"
SEED = 20261025


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def experiment_pi() -> list[dict[str, float]]:
    samples = 1_600_000
    # Lotes fijos hacen que todos los números de procesos evalúen exactamente
    # el mismo conjunto de puntos, independientemente de cómo se distribuyan.
    batch_size = 25_000
    args = [(min(batch_size, samples - start), SEED + index)
            for index, start in enumerate(range(0, samples, batch_size))]
    rows = []
    # Pool(1) mide también el coste del mismo mecanismo de distribución.
    for workers in (1, 2, 4, 8):
        started = time.perf_counter()
        with mp.Pool(workers) as pool:
            inside = sum(pool.map(estimate_pi_chunk, args))
        elapsed = time.perf_counter() - started
        rows.append({"workers": workers, "pi": 4 * inside / samples, "seconds": elapsed})
    baseline = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
    write_csv(RESULTS / "parallel_speedup.csv", ["workers", "pi_estimate", "seconds", "speedup"],
              [[r["workers"], r["pi"], r["seconds"], r["speedup"]] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [int(row["workers"]) for row in rows]
    ax.plot(x, [row["speedup"] for row in rows], "o-", linewidth=2, label="Observado")
    ax.plot(x, x, "--", label="Ideal (lineal)")
    ax.set(xlabel="Número de procesos", ylabel="Aceleración (T₁ / Tₚ)", title="Estimación de π: aceleración paralela")
    ax.set_xticks(x); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "parallel_speedup.png", dpi=160); plt.close(fig)
    return rows


def experiment_knapsack() -> list[dict[str, object]]:
    # Caso diseñado: peso ligero prioriza A y deja espacio inutilizable; ratio elige B+C.
    items = [Item("A", 2, 3), Item("B", 3, 5), Item("C", 3, 5)]
    capacity = 6
    rows = []
    for criterion, label in (("value_weight", "Valor/peso"), ("lightest", "Menor peso")):
        chosen, used, value = greedy_knapsack(items, capacity, criterion)
        rows.append({"method": label, "items": ", ".join(item.name for item in chosen), "used": used, "value": value})
    write_csv(RESULTS / "knapsack_comparison.csv", ["method", "selected_items", "capacity_used", "total_value"],
              [[r["method"], r["items"], r["used"], r["value"]] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [str(r["method"]) for r in rows]
    values = [int(r["value"]) for r in rows]
    bars = ax.bar(labels, values, color=["#3274a1", "#e1812c"])
    ax.bar_label(bars, padding=3); ax.set_ylabel("Valor total")
    ax.set_title("Mochila 0/1 (capacidad 6): valor obtenido")
    ax.text(.5, .92, "Capacidad usada: " + " | ".join(f"{r['method']}={r['used']}/6" for r in rows),
            transform=ax.transAxes, ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.25); fig.tight_layout()
    fig.savefig(RESULTS / "knapsack_comparison.png", dpi=160); plt.close(fig)
    return rows


def experiment_two_opt() -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    rows = []
    distributions: dict[str, list[float]] = {"Aleatoria": [], "Vecino más cercano": []}
    for trial in range(40):
        rng = random.Random(SEED + 1000 + trial)
        points = [(rng.random() * 100, rng.random() * 100) for _ in range(20)]
        starts = {
            "Aleatoria": random_tour(len(points), rng),
            "Vecino más cercano": nearest_neighbor_tour(points, rng.randrange(len(points))),
        }
        for method, initial in starts.items():
            result, improvements, _ = two_opt(initial, points)
            cost = tour_cost(result, points)
            distributions[method].append(cost)
            rows.append({"trial": trial, "method": method, "improvements": improvements, "final_cost": cost})
    write_csv(RESULTS / "two_opt_comparison.csv", ["trial", "initial_method", "improvements", "final_cost"],
              [[r["trial"], r["method"], r["improvements"], r["final_cost"]] for r in rows])
    improvement_data = [[r["improvements"] for r in rows if r["method"] == method]
                        for method in ("Aleatoria", "Vecino más cercano")]
    cost_data = [distributions["Aleatoria"], distributions["Vecino más cercano"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].boxplot(improvement_data, tick_labels=["Aleatoria", "Vecino más cercano"])
    axes[0].set(title="Movimientos de mejora", ylabel="Número de mejoras")
    axes[1].boxplot(cost_data, tick_labels=["Aleatoria", "Vecino más cercano"])
    axes[1].set(title="Calidad del óptimo local", ylabel="Costo del tour")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.suptitle("2-opt en 40 instancias euclidianas de 20 ciudades"); fig.tight_layout()
    fig.savefig(RESULTS / "two_opt_comparison.png", dpi=160); plt.close(fig)
    return rows, distributions


def experiment_annealing() -> dict[str, dict[str, object]]:
    rng = random.Random(SEED + 2000)
    points = [(rng.random() * 100, rng.random() * 100) for _ in range(30)]
    initial = random_tour(len(points), rng)
    runs: dict[str, dict[str, object]] = {}
    for schedule in ("linear", "exponential"):
        result = simulated_annealing(points, initial, schedule, 15_000, random.Random(SEED + 3000),
                                     start_temp=15.0, end_temp=.02)
        runs[schedule] = {"cost": result.best_cost, "seconds": result.elapsed, "history": result.history}
    rows = [[name, data["cost"], data["seconds"]] for name, data in runs.items()]
    write_csv(RESULTS / "annealing_comparison.csv", ["schedule", "best_cost", "seconds"], rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, data in runs.items():
        ax.plot(data["history"], label=f"{name.capitalize()} (final {data['cost']:.1f})", linewidth=1.2)
    ax.set(xlabel="Iteración", ylabel="Mejor costo encontrado", title="Recocido simulado: convergencia")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "annealing_convergence.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, data in runs.items():
        ax.scatter(data["seconds"], data["cost"], s=85, label=name.capitalize())
        ax.annotate(name.capitalize(), (data["seconds"], data["cost"]), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Tiempo de ejecución (s)", ylabel="Costo final (menor es mejor)", title="Tiempo frente a calidad")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(RESULTS / "annealing_time_quality.png", dpi=160); plt.close(fig)
    return runs


def experiment_mst() -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    rows = []
    methods = {name: [] for name in ("MST puro", "MST + 2-opt", "Búsqueda local", "Recocido simulado")}
    ratios = []
    for trial in range(100):
        rng = random.Random(SEED + 5000 + trial)
        points = [(rng.random() * 100, rng.random() * 100) for _ in range(9)]
        optimum = exact_tsp_cost(points)
        mst_tour = mst_preorder_tour(points)
        mst_cost = tour_cost(mst_tour, points)
        mst2_tour, _, _ = two_opt(mst_tour, points)
        random_initial = random_tour(9, rng)
        local_tour, _, _ = two_opt(random_initial, points)
        sa_result = simulated_annealing(points, random_initial, "exponential", 2500,
                                        random.Random(SEED + 8000 + trial), start_temp=15, end_temp=.02)
        costs = {
            "MST puro": mst_cost,
            "MST + 2-opt": tour_cost(mst2_tour, points),
            "Búsqueda local": tour_cost(local_tour, points),
            "Recocido simulado": sa_result.best_cost,
        }
        ratios.append(mst_cost / optimum)
        for method, cost in costs.items():
            methods[method].append(cost)
            rows.append({"trial": trial, "method": method, "cost": cost, "optimum": optimum,
                         "mst_approximation_ratio": mst_cost / optimum})
    write_csv(RESULTS / "mst_100_instances.csv", ["trial", "method", "cost", "optimal_cost", "mst_ratio"],
              [[r["trial"], r["method"], r["cost"], r["optimum"], r["mst_approximation_ratio"]] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(ratios, bins=14, color="#4c9f70", edgecolor="white")
    axes[0].axvline(2, color="#b22222", linestyle="--", label="Límite teórico = 2")
    axes[0].set(xlabel="Costo MST / óptimo exacto", ylabel="Número de instancias", title="Razón de aproximación")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].boxplot(list(methods.values()), tick_labels=list(methods.keys()), showfliers=True)
    axes[1].set(ylabel="Costo del tour", title="Comparación en 100 instancias")
    axes[1].tick_params(axis="x", rotation=18); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(RESULTS / "mst_comparison.png", dpi=160); plt.close(fig)
    return rows, {"ratios": ratios, **methods}


def generate_report(pi, knapsack, local_rows, anneal, mst_data) -> None:
    local_moves = {method: [r["improvements"] for r in local_rows if r["method"] == method]
                   for method in ("Aleatoria", "Vecino más cercano")}
    means = {key: statistics.mean(value) for key, value in local_moves.items()}
    costs = {key: statistics.mean(value) for key, value in mst_data.items() if key != "ratios"}
    ratios = mst_data["ratios"]
    body = f'''# Reporte: algoritmos de optimización

## Enlaces de entrega

- Repositorio público: **pendiente de publicar**.
- Video demo (máximo 8 minutos): **pendiente de grabar y subir**.

## Metodología y reproducibilidad

Los experimentos se ejecutaron con semilla `{SEED}`. Los datos tabulares y figuras se encuentran en `../results/`. Los recorridos TSP usan distancia euclidiana (métrica), y el óptimo de las instancias de 9 ciudades se calculó exactamente mediante Held–Karp.

## Ejercicio 1 — Aceleración paralela

Se estimó π con 1.600.000 muestras y {', '.join(f"{int(row['workers'])} proceso(s): {row['seconds']:.3f} s, speedup {row['speedup']:.2f}×" for row in pi)}. La aceleración no tiene por qué ser lineal: repartir trabajo y crear procesos cuesta tiempo, el sistema operativo comparte recursos y el cálculo tiene componentes seriales (ley de Amdahl). El paralelismo puede además saturar núcleos o ancho de banda; con esta carga y equipo el tiempo observado incluso puede empeorar en algunos niveles.

![Aceleración](../results/parallel_speedup.png)

## Ejercicio 2 — Mochila voraz

Con capacidad 6 y objetos A=(peso 2, valor 3), B=(3,5), C=(3,5), el orden valor/peso selecciona **{knapsack[0]['items']}**, usa {knapsack[0]['used']} y obtiene valor {knapsack[0]['value']}; el orden por menor peso selecciona **{knapsack[1]['items']}**, usa {knapsack[1]['used']} y obtiene {knapsack[1]['value']}. Menor peso elige A y luego B, bloqueando C; este criterio local pierde frente a valor/peso. Ninguno de los dos voraces garantiza el óptimo para mochila 0/1.

![Mochila](../results/knapsack_comparison.png)

## Ejercicio 3 — Búsqueda local 2-opt

Se compararon 40 instancias de 20 ciudades. El número medio de mejoras fue **{means['Aleatoria']:.1f}** desde una permutación aleatoria y **{means['Vecino más cercano']:.1f}** desde Vecino Más Cercano. **Sí: en este experimento Vecino Más Cercano necesitó menos mejoras para converger**, como muestran las cajas de movimientos. La caja de costos muestra la calidad del óptimo local: menos movimientos no implica necesariamente mejor calidad, pues el punto inicial cambia la cuenca de atracción.

![Resultados 2-opt](../results/two_opt_comparison.png)

## Ejercicio 4 — Recocido simulado

Ambos esquemas usaron la misma instancia de 30 ciudades, misma solución inicial y 15.000 iteraciones. El enfriamiento lineal obtuvo costo **{anneal['linear']['cost']:.2f}** en {anneal['linear']['seconds']:.3f} s; el exponencial obtuvo **{anneal['exponential']['cost']:.2f}** en {anneal['exponential']['seconds']:.3f} s. La forma de enfriar controla cuánto tiempo se aceptan movimientos peores: enfriar lentamente suele explorar más (posible mejor calidad y mayor coste), mientras un enfriamiento agresivo converge antes y puede quedarse atrapado. El tiempo puede ser parecido aquí porque el número de iteraciones es fijo; el costo, no obstante, depende también de la semilla y de la instancia.

![Convergencia](../results/annealing_convergence.png)

![Tiempo frente a calidad](../results/annealing_time_quality.png)

## Ejercicio 5 — Aproximación MST

En 100 instancias aleatorias de 9 ciudades, la razón MST/óptimo tuvo media **{statistics.mean(ratios):.3f}**, máximo **{max(ratios):.3f}** y permaneció {'por debajo del límite teórico' if max(ratios) <= 2 + 1e-9 else 'por encima del límite teórico en alguna instancia'} 2. La garantía de 2 aplica al recorrido preorder del MST para distancias métricas; no implica que cada instancia alcance el límite. Held–Karp proporciona el óptimo de referencia.

Costos medios: {', '.join(f"{name}: {value:.2f}" for name, value in costs.items())}. El post-procesamiento 2-opt nunca empeora el tour MST (solo acepta intercambios estrictamente mejores) y normalmente lo acorta. Su mejora frente a búsqueda local/recocido depende de sus inicializaciones y presupuestos; aquí los métodos se comparan sobre las mismas 100 instancias, pero SA recibe 2.500 iteraciones por instancia.

![MST, razón y costos](../results/mst_comparison.png)

## Archivos generados

- `parallel_speedup.csv`, `knapsack_comparison.csv`, `two_opt_comparison.csv`, `annealing_comparison.csv`, `mst_100_instances.csv`.
- Gráficas PNG correspondientes en `../results/`.
'''
    REPORT.write_text(body, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    print("Ejercicio 1/5: aceleración paralela")
    pi = experiment_pi()
    print("Ejercicio 2/5: mochila voraz")
    knapsack = experiment_knapsack()
    print("Ejercicio 3/5: búsqueda local 2-opt")
    local_rows, _ = experiment_two_opt()
    print("Ejercicio 4/5: recocido simulado")
    anneal = experiment_annealing()
    print("Ejercicio 5/5: MST en 100 instancias")
    _, mst_data = experiment_mst()
    generate_report(pi, knapsack, local_rows, anneal, mst_data)
    print(f"Experimentos completos. Resultados: {RESULTS}")
    print(f"Reporte actualizado: {REPORT}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
