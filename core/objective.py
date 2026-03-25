def compute_slack(delay, deadline):
    """
    Corresponds to the deadline violation slack idea:
    slack = max(0, delay - deadline)
    """
    return max(0.0, delay - deadline)


def compute_objective(delay, deadline, beta):
    """
    A simple objective form for validating formula (1):
    objective = delay + beta * slack
    """
    slack = compute_slack(delay, deadline)
    return delay + beta * slack