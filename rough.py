import json, re
registry = json.load(open("data/processed/ground_truth_registry.json"))
from collections import Counter
conditions = Counter(re.search(r"_(RC|Train|Dyn|DynBigram)", k).group(1) if re.search(r"_(RC|Train|Dyn|DynBigram)", k) else "?" for k in registry)
print(conditions)

rc_words = [(k,v) for k,v in registry.items() if "_RC" in k]
from collections import Counter
print(Counter(v for k,v in rc_words).most_common(5))

rc_sample = [(k,v) for k,v in registry.items() if "_RC" in k][:3]
print(rc_sample)

print(len(registry))  # expect ~307 if RC+Dyn both already in