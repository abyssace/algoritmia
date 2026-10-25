"""Implementaciones autocontenidas para los ejercicios del proyecto."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

Point = tuple[float, float]


def estimate_pi_chunk(args: tuple[int, int]) -> int:
    """Cuenta puntos dentro del círculo para una partición reproducible."""
    samples, seed = args
    rng = random.Random(seed)
    return sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))


@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int


def greedy_knapsack(items: Sequence[Item], capacity: int, criterion: str) -> tuple[list[Item], int, int]:
    """Mochila fraccionaria no; mochila 0/1 voraz ordenada por un criterio."""
    if capacity < 0:
        raise ValueError("La capacidad no puede ser negativa")
    if criterion == "value_weight":
        ordered = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
    elif criterion == "lightest":
        ordered = sorted(items, key=lambda item: item.weight)
    else:
        raise ValueError("criterion debe ser 'value_weight' o 'lightest'")
    chosen: list[Item] = []
    used = total_value = 0
    for item in ordered:
        if item.weight <= 0:
            raise ValueError("Los pesos deben ser positivos")
        if used + item.weight <= capacity:
            chosen.append(item)
            used += item.weight
            total_value += item.value
    return chosen, used, total_value


def tour_cost(tour: Sequence[int], points: Sequence[Point]) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def random_tour(n: int, rng: random.Random) -> list[int]:
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def nearest_neighbor_tour(points: Sequence[Point], start: int) -> list[int]:
    unvisited = set(range(len(points)))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        current = tour[-1]
        nxt = min(unvisited, key=lambda city: math.dist(points[current], points[city]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def two_opt(tour: Sequence[int], points: Sequence[Point]) -> tuple[list[int], int, list[float]]:
    """2-opt first-improvement hasta óptimo local; devuelve movimientos y trayectoria."""
    best = list(tour)
    n = len(best)
    if n < 4:
        return best, 0, [tour_cost(best, points)]
    cost = tour_cost(best, points)
    history = [cost]
    improvements = 0
    while True:
        improved = False
        for i, k in combinations(range(n), 2):
            if i == 0 and k == n - 1:
                continue  # invertir el ciclo completo no cambia sus aristas
            a, b = best[i], best[(i + 1) % n]
            c, d = best[k], best[(k + 1) % n]
            delta = math.dist(points[a], points[c]) + math.dist(points[b], points[d]) \
                - math.dist(points[a], points[b]) - math.dist(points[c], points[d])
            if delta < -1e-12:
                best[i + 1:k + 1] = reversed(best[i + 1:k + 1])
                cost += delta
                improvements += 1
                history.append(cost)
                improved = True
                break
        if not improved:
            return best, improvements, history


def mst_preorder_tour(points: Sequence[Point]) -> list[int]:
    """Tour métrico doblemente aproximado por recorrido preorder del MST (2-approx)."""
    n = len(points)
    if n == 0:
        return []
    in_tree = {0}
    edges: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    while len(in_tree) < n:
        weight, parent, child = min(
            (math.dist(points[u], points[v]), u, v)
            for u in in_tree for v in range(n) if v not in in_tree
        )
        in_tree.add(child)
        edges[parent].append((child, weight))
        edges[child].append((parent, weight))
    order: list[int] = []
    seen = set()

    def visit(node: int) -> None:
        seen.add(node)
        order.append(node)
        for child, _ in sorted(edges[node], key=lambda edge: edge[1]):
            if child not in seen:
                visit(child)

    visit(0)
    return order


def exact_tsp_cost(points: Sequence[Point]) -> float:
    """Óptimo exacto por Held–Karp, fijando la ciudad 0 para evitar simetrías."""
    n = len(points)
    if n <= 1:
        return 0.0
    # dp[(mask, last)] es el mínimo camino 0 -> ... -> last que cubre mask (ciudades 1..n-1).
    dp: dict[tuple[int, int], float] = {}
    for city in range(1, n):
        dp[(1 << (city - 1), city)] = math.dist(points[0], points[city])
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            mask = sum(1 << (city - 1) for city in subset)
            for last in subset:
                prev_mask = mask ^ (1 << (last - 1))
                dp[(mask, last)] = min(
                    dp[(prev_mask, prev)] + math.dist(points[prev], points[last])
                    for prev in subset if prev != last
                )
    full = (1 << (n - 1)) - 1
    return min(dp[(full, last)] + math.dist(points[last], points[0]) for last in range(1, n))


@dataclass
class AnnealingResult:
    best_tour: list[int]
    best_cost: float
    elapsed: float
    history: list[float]


def simulated_annealing(
    points: Sequence[Point], initial: Sequence[int], schedule: str, iterations: int,
    rng: random.Random, start_temp: float = 12.0, end_temp: float = 0.02,
) -> AnnealingResult:
    """Recocido simulado con movimiento 2-opt aleatorio."""
    import time

    if schedule not in {"linear", "exponential"}:
        raise ValueError("schedule debe ser 'linear' o 'exponential'")
    current = list(initial)
    current_cost = tour_cost(current, points)
    best, best_cost = current[:], current_cost
    history = [best_cost]
    n = len(current)
    started = time.perf_counter()
    for iteration in range(iterations):
        fraction = iteration / max(1, iterations - 1)
        if schedule == "linear":
            temperature = start_temp + (end_temp - start_temp) * fraction
        else:
            temperature = start_temp * (end_temp / start_temp) ** fraction
        i, k = sorted(rng.sample(range(n), 2))
        if i == 0 and k == n - 1:
            k -= 1
        candidate = current[:]
        candidate[i + 1:k + 1] = reversed(candidate[i + 1:k + 1])
        candidate_cost = tour_cost(candidate, points)
        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-15)):
            current, current_cost = candidate, candidate_cost
            if current_cost < best_cost:
                best, best_cost = current[:], current_cost
        history.append(best_cost)
    return AnnealingResult(best, best_cost, time.perf_counter() - started, history)
