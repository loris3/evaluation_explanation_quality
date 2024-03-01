A collection of experiments to assess the quality of explanations for detectors of machine-generated text.
# Setup

**Cloning**

`git clone --recurse-submodules repo_url`

`cd repo_dir`

**Models**

`cd repo_dir`

`wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt models/radford et al/detector-base.pt`

This detector requires an older version of transformers.

`cd transformers`

`git checkout v2.1.1`

All other modules use a newer version.

**Cache**

Unzip `explanation_cache.zip` to `explanation_cache`. The filenames contain the SHA256 hash of the input string. See `fi_explainer.py`.
# Detectors
All detectors are extended to support masked input:
- In-domain fine-tuned RoBERTa in [Guo et al. 2023](https://arxiv.org/abs/2301.07597): [detector_guo.py](./detector_guo.py)
- Out-of-domain fine-tuned RoBERTa of [Radford et al. 2018](https://github.com/openai/gpt-2-output-dataset): [gpt2outputdataset/detector_radford.py](./gpt2outputdataset/detector_radford.py)
- Zero-shot method in [Mitchell et al. 2023](https://arxiv.org/abs/2301.11305v1) with suggestions in [Mireshghallah et al. 2023](https://arxiv.org/abs/2305.09859): [detector_detectgpt.py](./detector_detectgpt.py)
# Explanation Methods
SHAP is used as-is.

Forks of LIME and Anchor are provided as submodules. 
## Anchor
- Addition of a *budget* limiting the number of samples used during search to cap runtime (200 samples per candidate during search, unlimited samples in final "best of each size" round)
- DistillBERT was replaced with DistillRoBERTA and the mask probability adjusted to increase coherence of perturbations
- Changes to the rendering functions for the user-study (used to share JS and CSS scope with LIME)
## LIME
- Cosmetic changes to the bar-charts for the user-study

# Experiments
The explanations are provided as a zip file. All experiments are designed so that any subset of the dataset can be processed in parallel by executing the notebooks with different offsets. 