# COMP-3821-SIMULATOR-FILE
Python file for our simulator to conduct non-clairevoyant scheduler testing

# grr vs spt simulator
# - alg_total: exact grr total completion time (round-by-round)
# - opt_total: spt closed form
# - worst_ratio: adversarial search for "hard" sizes S_{l-1}+1, descending order
# - brute_force_check: exhaustive check to validate that restriction
# - grid: sweep (b0, alpha) and report the params with the smallest worst-case ratio
# - plot_results: heatmap of sweep + simulated ratio vs proven upper bound
# running the file saves the diagram to Desktop, named by timestamp
