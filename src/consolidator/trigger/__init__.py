"""consolidator-trigger — event-log consumer that fires the
consolidator's HTTP dispatch endpoint per event.

Lives in this repo (rather than its own) because it tracks the
dispatch contract one-to-one and ships in the same release.
"""
