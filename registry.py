import yaml

VALID_FREQUENCIES = {
    "15m", "hourly", "daily", "weekly", "monthly", "event", "multi-frequency"
}

class StreamRegistry:
    """Load and validate data stream configuration."""

    def __init__(self, config_path="config/data_streams.yaml"):
        self.streams = self._load(config_path)

    def _load(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or []

        seen = set()
        streams = {}
        for entry in data:
            name = entry.get("name")
            freq = entry.get("frequency")
            if not name or not freq:
                raise ValueError(f"Stream entry missing required fields: {entry}")
            if name in seen:
                raise ValueError(f"Duplicate stream name: {name}")
            if freq not in VALID_FREQUENCIES:
                raise ValueError(f"Invalid frequency '{freq}' for stream {name}")
            seen.add(name)
            streams[name] = entry
        return streams

    def get(self, name):
        return self.streams.get(name)

    def approved_streams(self):
        return {
            n: s for n, s in self.streams.items() if s.get("approved", False)
        }
