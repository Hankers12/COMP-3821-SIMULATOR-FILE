# grr vs spt simulator
# - alg_total: exact grr total completion time (round-by-round)
# - opt_total: spt closed form
# - worst_ratio: adversarial search for "hard" sizes S_{l-1}+1, descending order
# - brute_force_check: exhaustive check to validate that restriction
# - grid: sweep (b0, alpha) and report the params with the smallest worst-case ratio
# - plot_results: heatmap of sweep + simulated ratio vs proven upper bound
# running the file saves the diagram to Desktop, named by timestamp

import math, random, itertools, os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------- algorithm + opt ----------

def slices(b_0, alpha, P):
    # slice lengths b_r and running totals S_r, until S_r >= P (lemma 3)
    b = []
    S = []
    total = 0
    r = 0

    while total < P:
        b_r = math.ceil(b_0 * (alpha ** r) - 1e-9)  # 1e-9 to guard float error against ceil, i.e. so 3.000000001 doesnt get rounded to 4
        b.append(b_r)
        total += b_r
        S.append(total)     # S[r] = b_0 + b_1 + ... + b_r
        r += 1

    return b, S


def grr_completion_times(job_sizes, load_cost, initial_slice, growth_factor) -> list:
    # completion time C_i of every job under geometric round-robin
    # follows "Geometric round-robin algorithm" steps from doc, timing rules from def 3
    # job_sizes visited in this order each round (adversary's labelling, see "Job ordering")
    # returns list where entry i is C_i

    # slices only needed up to level of largest job (lemma 3)
    slice_lengths, _ = slices(initial_slice, growth_factor, max(job_sizes))

    clock = 0                                   # start s_k of next allocation
    work_done = [0] * len(job_sizes)            # w_i work done so far
    completion_time = [None] * len(job_sizes)
    unfinished = list(range(len(job_sizes)))

    for slice_length in slice_lengths:          # round r gets slice b_r
        still_unfinished = []

        for job in unfinished:
            remaining_work = job_sizes[job] - work_done[job]
            allocation_start = clock

            # allocation occupies [s, s + c + b_r) even if job finishes early, processor idles for rest of slice
            clock += load_cost + slice_length

            if remaining_work <= slice_length:
                # def 3, C_i = s_k + c + (p_i - w_i)
                completion_time[job] = allocation_start + load_cost + remaining_work
                continue

            work_done[job] += slice_length
            still_unfinished.append(job)

        # finished jobs are skipped in later rounds and take no time
        unfinished = still_unfinished

    # lemma 3, every job is done by round l(max p), the last slice computed
    assert not unfinished, "slices() did not cover the largest job"
    return completion_time


def alg_total(job_sizes, load_cost, initial_slice, growth_factor):
    # total completion time sum_i C_i of geometric round-robin
    return sum(grr_completion_times(job_sizes, load_cost, initial_slice, growth_factor))


def opt_total(sizes, load_cost):
    # spt (def 6), shortest job first, one allocation each, eq (3)
    clock = 0
    total_comp_time = 0
    for size in sorted(sizes):
        clock += load_cost + size
        total_comp_time += clock
    return total_comp_time


def ratio(sizes, c, b0, alpha):
    # ALG / OPT for one instance, the thing inside the sup in def 5
    return alg_total(sizes, c, b0, alpha) / opt_total(sizes, c)


# ---------- adversary search ----------

def hard_sizes(initial_slice, growth_factor, max_size):
    # worst case job size in each level, largest first S_{l-1}+1 (max idle overshoot)
    _, service_totals = slices(initial_slice, growth_factor, max_size)

    level_zero_size = 1             # S_{-1} = 0, so level 0 starts at size 1
    later_level_sizes = {total + 1 for total in service_totals if total + 1 <= max_size}

    return sorted({level_zero_size} | later_level_sizes, reverse = True)


def build_instance(level_sizes, jobs_per_level):
    # turn per-level job counts into a list of job sizes
    # level_sizes is descending, so instance comes in descending order (lemma 9)
    job_sizes = []
    for size, count in zip(level_sizes, jobs_per_level):
        job_sizes += [size] * count
    return job_sizes


def counts_ratio(level_sizes, jobs_per_level, params):
    # ALG/OPT for instance with jobs_per_level[i] jobs of size level_sizes[i]
    # params is (load_cost, initial_slice, growth_factor), empty instance scores 0
    job_sizes = build_instance(level_sizes, jobs_per_level)
    if not job_sizes:
        return 0
    return ratio(job_sizes, *params)


def single_level_counts(level, num_levels, num_jobs):
    # every job placed at one level (the equal-jobs instance)
    counts = [0] * num_levels
    counts[level] = num_jobs
    return counts


def starting_points(num_levels, num_jobs):
    # starting count vectors: all jobs at one level (one per level) + an even spread
    single_level_starts = [single_level_counts(level, num_levels, num_jobs)
                           for level in range(num_levels)]
    even_spread = [num_jobs // num_levels] * num_levels
    return single_level_starts + [even_spread]


def best_starting_point(level_sizes, num_jobs, params):
    # (ratio, counts) for best of the starting vectors
    starts = starting_points(len(level_sizes), num_jobs)
    return max((counts_ratio(level_sizes, counts, params), counts) for counts in starts)


def random_move(counts, rng, max_jobs_moved):
    # copy of counts with some jobs moved from one random level to another
    # rng only picks the move, job sizes are never random
    source_level = rng.randrange(len(counts))
    target_level = rng.randrange(len(counts))
    jobs_moved = min(counts[source_level], rng.randint(1, max_jobs_moved))

    candidate = counts.copy()
    candidate[source_level] -= jobs_moved
    candidate[target_level] += jobs_moved
    return candidate


def hill_climb(level_sizes, start, params, iterations, max_jobs_moved, seed):
    # try random moves from start, keep any that raises ALG/OPT
    # fixed seed so runs are reproducible, returns best (ratio, counts)
    rng = random.Random(seed)
    best_ratio, best_counts = start

    for _ in range(iterations):
        candidate = random_move(best_counts, rng, max_jobs_moved)
        candidate_ratio = counts_ratio(level_sizes, candidate, params)
        if candidate_ratio > best_ratio:
            best_ratio, best_counts = candidate_ratio, candidate

    return best_ratio, best_counts


def worst_ratio(load_cost, initial_slice, growth_factor, max_size, num_jobs = 300, iterations = 1500, seed = 0):
    # largest ALG/OPT found for these params, approximating the sup in def 5
    # only searches number of jobs per level (hard_sizes, descending order)
    # found by a real instance so its a LOWER bound on the true competitive ratio
    # num_jobs large to approximate n -> infinity, where worst cases occur
    # returns (best ratio found, {job size: number of jobs at that size})
    params = (load_cost, initial_slice, growth_factor)
    level_sizes = hard_sizes(initial_slice, growth_factor, max_size)
    max_jobs_moved = max(1, num_jobs // 8)

    start = best_starting_point(level_sizes, num_jobs, params)
    best_ratio, best_counts = hill_climb(level_sizes, start, params, iterations, max_jobs_moved, seed)

    return best_ratio, dict(zip(level_sizes, best_counts))


#  checks + sweep 

def all_multisets(candidate_sizes, max_jobs):
    # every multiset of 1..max_jobs jobs whose sizes come from candidate_sizes
    for num_jobs in range(1, max_jobs + 1):
        yield from itertools.combinations_with_replacement(candidate_sizes, num_jobs)


def brute_force_check(load_cost = 2, initial_slice = 2, growth_factor = 2,
                      max_size = 12, max_jobs = 4):
    # checks hard_sizes in descending order loses nothing vs trying everything
    # exhaustive: every multiset from {1..P} (def 1), every order ("Job ordering")
    # defaults = core setting b0 = c, alpha = 2, scaled to c = 2
    # returns (exhaustive worst, restricted worst), should match
    params = (load_cost, initial_slice, growth_factor)

    exhaustive_worst = 0
    for multiset in all_multisets(range(1, max_size + 1), max_jobs):
        for order in set(itertools.permutations(multiset)):
            exhaustive_worst = max(exhaustive_worst, ratio(list(order), *params))

    restricted_worst = 0
    for multiset in all_multisets(hard_sizes(initial_slice, growth_factor, max_size), max_jobs):
        descending = sorted(multiset, reverse = True)
        restricted_worst = max(restricted_worst, ratio(descending, *params))

    return exhaustive_worst, restricted_worst


def grid(load_cost = 10, max_size = 10**5, slice_ratios = None, growth_factors = None, num_jobs = 300) -> dict[tuple[float, float], float]:
    # worst ratio found for each (b0/c, alpha) pair, prints each + best pair
    # (proposal question: how ratio depends on alpha and initial slice)
    # uses beta = b0/c since only the ratio matters (scale invariance)
    # b0 = beta * c must be a positive int, so pick load_cost big enough (c = 10 allows 0.5, 1.5)
    # returns {(beta, alpha): worst ratio found}
    slice_ratios = slice_ratios or [0.5, 1, 1.5, 2, 2.5, 3, 4]
    growth_factors = growth_factors or [1.25, 1.5, 1.75, 2, 2.5, 3, 4]

    results: dict[tuple[float, float], float] = {}     # (beta, alpha) -> worst ratio
    for beta in slice_ratios:
        initial_slice = round(beta * load_cost)
        if initial_slice < 1 or initial_slice != beta * load_cost:
            raise ValueError(f"b0 = {beta} * {load_cost} is not a positive integer; increase load_cost")

        for alpha in growth_factors:
            worst, _ = worst_ratio(load_cost, initial_slice, alpha, max_size, num_jobs = num_jobs)
            results[(beta, alpha)] = worst
            print(f"b0/c={beta:<4} alpha={alpha:<5} worst ratio ~ {worst:.3f}")

    best_beta, best_alpha = min(results, key = lambda params: results[params])
    print(f"\nBest on grid: b0/c={best_beta}, alpha={best_alpha}, "
          f"worst ratio ~ {results[(best_beta, best_alpha)]:.3f}")
    return results


# plotting 

def upper_bound(beta, alpha):
    # proven bound rho <= 2 * max(alpha + 1/beta, 1 + beta), lemma 6 generalised to b0 = beta * c
    # gives 6 at core (beta = 1, alpha = 2), ignores ceilings
    return 2 * max(alpha + 1 / beta, 1 + beta)


def grid_axes(results):
    # sorted betas and alphas used in grid() results
    betas = sorted({beta for beta, _ in results})
    alphas = sorted({alpha for _, alpha in results})
    return betas, alphas


def results_matrix(results, betas, alphas):
    # results as 2d list, row per beta, column per alpha
    return [[results[(beta, alpha)] for alpha in alphas] for beta in betas]


def outline_cell(ax, row, col, colour, label):
    # coloured box around one heatmap cell
    box = Rectangle((col - 0.5, row - 0.5), 1, 1, fill = False,
                    edgecolor = colour, linewidth = 3, label = label)
    ax.add_patch(box)


def plot_heatmap(ax, results):
    # worst ratio found for each (b0/c, alpha), lower is better
    # outlines best cell and core setting (b0 = c, alpha = 2)
    betas, alphas = grid_axes(results)
    matrix = results_matrix(results, betas, alphas)

    image = ax.imshow(matrix, cmap = "viridis_r", aspect = "auto", origin = "lower")
    plt.colorbar(image, ax = ax, label = "worst ALG/OPT found")

    midpoint = (min(results.values()) + max(results.values())) / 2
    for row, beta in enumerate(betas):
        for col, alpha in enumerate(alphas):
            value = results[(beta, alpha)]
            text_colour = "black" if value < midpoint else "white"   # readable on light/dark cells
            ax.text(col, row, f"{value:.2f}", ha = "center", va = "center",
                    fontsize = 8, color = text_colour)

    best_beta, best_alpha = min(results, key = lambda params: results[params])
    outline_cell(ax, betas.index(best_beta), alphas.index(best_alpha), "red", "best found")
    if (1, 2) in results:
        outline_cell(ax, betas.index(1), alphas.index(2), "orange", "core (b0 = c, alpha = 2)")

    ax.set_xticks(range(len(alphas)), [str(a) for a in alphas])
    ax.set_yticks(range(len(betas)), [str(b) for b in betas])
    ax.set_xlabel("growth factor alpha")
    ax.set_ylabel("initial slice b0 / c")
    ax.set_title("Worst-case ratio found by the simulator")
    ax.legend(loc = "upper center", bbox_to_anchor = (0.5, -0.12), ncol = 2, fontsize = 8)


def best_over_beta(values_at, betas, alphas):
    # for each alpha, smallest value over all betas (best b0 for that alpha)
    return [min(values_at(beta, alpha) for beta in betas) for alpha in alphas]


def plot_simulated_vs_bound(ax, results):
    # for each alpha, best ratio over b0/c: simulated (lower bound) vs proven upper bound
    # true best ratio lies between the two lines
    betas, alphas = grid_axes(results)
    simulated = best_over_beta(lambda b, a: results[(b, a)], betas, alphas)
    proven = best_over_beta(upper_bound, betas, alphas)

    ax.plot(alphas, simulated, "o-", label = "simulated (lower bound on rho)")
    ax.plot(alphas, proven, "s--", label = "proven upper bound 2max(alpha+1/beta, 1+beta), ignoring ceilings")
    ax.fill_between(alphas, simulated, proven, alpha = 0.15, label = "true rho lies in here")

    ax.set_xlabel("growth factor alpha")
    ax.set_ylabel("best ratio over b0/c")
    ax.set_title("Best b0 for each alpha: simulation vs proof")
    ax.legend(fontsize = 8)
    ax.grid(True, alpha = 0.3)


def desktop_save_path():
    # ~/Desktop/grr_sweep_<timestamp>.png, falls back to current folder if no Desktop
    # (on windows with onedrive the desktop can be somewhere else, so check it exists)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    folder = desktop if os.path.isdir(desktop) else os.getcwd()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(folder, f"grr_sweep_{timestamp}.png")


def plot_results(results, filename, show = False):
    # heatmap + simulated vs bound side by side, saved to filename
    fig, (heatmap_ax, line_ax) = plt.subplots(1, 2, figsize = (14, 5.5))
    plot_heatmap(heatmap_ax, results)
    plot_simulated_vs_bound(line_ax, results)

    fig.tight_layout()
    fig.savefig(filename, dpi = 150)
    print(f"Saved plot to {filename}")
    if show:
        plt.show()


if __name__ == "__main__":
    print("Brute-force vs hard-size search (tiny instances):", brute_force_check())

    core_ratio, core_instance = worst_ratio(load_cost = 10, initial_slice = 10,
                                            growth_factor = 2, max_size = 10**5)
    print("Core params b0 = c, alpha = 2:", core_ratio)

    results = grid()
    plot_results(results, desktop_save_path())