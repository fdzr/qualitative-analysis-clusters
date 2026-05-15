from collections import namedtuple

Use = namedtuple("Uses", ["word", "text", "start", "end"])

ShortUse = namedtuple("ShorUses", ["word", "id"])

Results = namedtuple("Results", ["jsd", "cluster_to_freq1", "cluster_to_freq2"])
