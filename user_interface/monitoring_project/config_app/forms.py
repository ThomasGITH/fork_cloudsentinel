from django import forms


def create_dynamic_form(config_data):
    class ConfigForm(forms.Form):
        pass

    for key, value in config_data.items():
        field_type = forms.CharField  # Default to CharField
        if isinstance(value, bool):
            field_type = forms.BooleanField
        elif isinstance(value, int):
            field_type = forms.IntegerField
        elif isinstance(value, float):
            field_type = forms.FloatField

        field = field_type(initial=value, required=False)
        ConfigForm.base_fields[key] = field

    return ConfigForm


class FileUploadForm(forms.Form):
    file = forms.FileField()


class MonitoringForm(forms.Form):
    display_name = forms.CharField(
        label="Session name",
        max_length=120,
        required=False,
        help_text="Optional recognizable name for Monitoring History.",
    )
    model_id = forms.ChoiceField(choices=[], label="Saved model artefact", required=True)
    window_minutes = forms.IntegerField(min_value=1, max_value=1440, initial=10, help_text="Amount of recent Prometheus data retained per detection cycle.")
    poll_interval_seconds = forms.IntegerField(min_value=5, max_value=3600, initial=300, help_text="How often CloudSentinel collects and evaluates a new window.")
    automatic_rca_enabled = forms.BooleanField(
        label="Enable automatic CausalRCA",
        required=False,
        initial=False,
        help_text="Optional. Monitoring remains available when this model input is not RCA-compatible.",
    )
    rca_threshold_percentage = forms.FloatField(
        label="RCA anomaly threshold (%)",
        min_value=0,
        max_value=100,
        initial=50,
        required=False,
        help_text="A run starts only when the newly evaluated cycle is strictly above this value.",
    )
    rca_analysis_window_minutes = forms.IntegerField(
        label="RCA analysis window (minutes)", min_value=1, max_value=1440, initial=10, required=False
    )
    rca_cooldown_minutes = forms.IntegerField(
        label="RCA cooldown (minutes)", min_value=0, max_value=10080, initial=60, required=False
    )
    rca_reanalysis_enabled = forms.BooleanField(
        label="Allow bounded RCA reanalysis within the same episode",
        required=False,
        initial=False,
        help_text="Off by default. When enabled, later high cycles may be analysed after all interval, new-data and budget gates pass.",
    )
    rca_reanalysis_min_interval_seconds = forms.IntegerField(
        label="Minimum RCA reanalysis interval (seconds)",
        min_value=60,
        max_value=604800,
        initial=900,
        required=False,
        help_text="Technical default: 900 seconds. This is separate from cooldown between episodes.",
    )
    rca_reanalysis_min_new_observations = forms.IntegerField(
        label="Minimum new observations for RCA reanalysis",
        min_value=1,
        max_value=100000,
        initial=20,
        required=False,
    )
    rca_max_attempts_per_episode = forms.IntegerField(
        label="Maximum RCA execution attempts per episode",
        min_value=1,
        max_value=20,
        initial=3,
        required=False,
        help_text="Includes the first accepted execution. Preflight skips do not consume this budget.",
    )


class UploadCGNNTrainDataForm(forms.Form):
    train_array = forms.FileField(label='Train Array')
    test_array = forms.FileField(label='Test Array')
    anomaly_label_array = forms.FileField(label='Anomaly Label Array')
    anomaly_sequence = forms.BooleanField(label='Does this contain an Anomaly Sequence?', required=False)
    dataset = forms.CharField(label='Name Your Dataset', max_length=100)
    comment = forms.CharField(label='Comment', max_length=255, required=False)
    metrics = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        label='Select Metrics'
    )
    ordered_metrics = forms.CharField(widget=forms.HiddenInput(), required=False)
