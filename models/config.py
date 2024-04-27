"""
Global model configuration parameters
"""

feature_selection = {
    # controls which features are used by which models #
    "src_n_transactions": set(),
    "dst_n_transactions": set(),
    "src_min_amt": set(),
    "src_mean_amt": set(),
    "src_median_amt": set(),
    "src_max_amt": set(),
    "src_ratio_to_min_amt": set(),
    "src_ratio_to_mean_amt": set(),
    "src_ratio_to_median_amt": set(),
    "src_ratio_to_max_amt": set(),
    "dst_min_amt": set(),
    "dst_mean_amt": set(),
    "dst_median_amt": set(),
    "dst_max_amt": set(),
    "dst_ratio_to_min_amt": set(),
    "dst_ratio_to_mean_amt": set(),
    "dst_ratio_to_median_amt": set(),
    "dst_ratio_to_max_amt": set(),
    "src_n_transactions_this_day_of_week": set(),
    "src_prop_transactions_this_day_of_week": set(),
    "dst_n_transactions_this_day_of_week": set(),
    "dst_prop_transactions_this_day_of_week": set(),
    "src_n_transactions_this_time": set(),
    "src_prop_transactions_this_time": set(),
    "dst_n_transactions_this_time": set(),
    "dst_prop_transactions_this_time": set(),
    "src_n_transactions_this_dst": set(),
    "src_prop_transactions_this_dst": set(),
    "dst_n_transactions_this_src": set(),
    "dst_prop_transactions_this_src": set(),
}
