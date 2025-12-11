from typing import Dict

# Simple pipeline combinator utilities
class Pipeline(object):
    """Common pipeline class for composing generator-based tasks."""

    def __init__(self, source=None):
        self.source = source

    def __iter__(self):
        return self.generator()

    def generator(self):
        """Yield pipeline data from the upstream source."""
        while self.has_next():
            try:
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __or__(self, other):
        """Allows connecting pipeline tasks using the | operator."""
        if other is not None:
            other.source = self.generator()
            return other
        return self

    def filter(self, data: Dict):
        """Override to filter out pipeline data."""
        return True

    def map(self, data: Dict):
        """Override to transform pipeline data."""
        return data

    def has_next(self) -> bool:
        """Override to stop the generator in certain conditions."""
        return True
