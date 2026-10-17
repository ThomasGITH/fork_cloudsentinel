"""Shared UI vocabulary for user-supplied workload context metadata."""

WORKLOAD_INTENSITIES = ("Low", "Normal", "High", "Variable")
WORKLOAD_INTENSITY_DEFINITIONS = (
    ("Low", "Sustained load clearly below the system's usual operating level."),
    ("Normal", "Representative of ordinary production operation."),
    ("High", "Sustained load materially above the system's usual operating level."),
    ("Variable", "Substantial changes in load occur within the dataset."),
)
WORKLOAD_CHARACTERISTICS = (
    "CPU-intensive",
    "Memory-intensive",
    "High traffic",
    "Latency-intensive",
    "Mixed",
)
ANOMALY_SCENARIOS = (
    "Normal operation",
    "CPU stress",
    "Memory stress",
    "Network delay",
    "Unknown",
    "Other",
)
INCIDENT_SCENARIOS = tuple(item for item in ANOMALY_SCENARIOS if item != "Other")
ANOMALY_SCENARIO_DETAIL_MAX_LENGTH = 160

DOMINANT_CHARACTERISTIC_HELP = (
    "The main resource or traffic characteristic in this dataset, such as CPU-intensive, "
    "Memory-intensive, High traffic, Latency-intensive or Mixed."
)
ANOMALY_SCENARIO_HELP = (
    "The main expected or observed scenario in this dataset. CloudSentinel uses it to "
    "analyse evaluations by context."
)
USER_SUPPLIED_HELP = (
    "Workload context is supplied by you; CloudSentinel does not infer these categories."
)


def scenario_display(workload_context):
    context = workload_context if isinstance(workload_context, dict) else {}
    scenario = context.get("anomaly_scenario")
    detail = context.get("anomaly_scenario_detail")
    if scenario == "Other" and detail:
        return f"Other — {detail}"
    return scenario
