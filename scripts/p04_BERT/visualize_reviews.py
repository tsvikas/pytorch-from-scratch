# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

from pytorch_from_scratch.p04_BERT.get_reviews import get_reviews

# %%
reviews = get_reviews()
df = pd.DataFrame(reviews)
# %%
df.text.str.len().plot.hist(bins=100, title="Review length")
# %%
df.stars.value_counts().reindex(range(1, 11)).plot.bar(title="Stars")
# %%
