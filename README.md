A collection of experiments to assess the quality of explanations for detectors of machine-generated text.

`git clone --recurse-submodules`
# Detectors
All detectors 
- In-domain fine-tuned RoBERTa in [Guo et al. 2023](https://arxiv.org/abs/2301.07597): [detector_guo.py](./detector_guo.py)
- Out-of-domain fine-tuned RoBERTa of [Radford et al. 2018](https://github.com/openai/gpt-2-output-dataset): [gpt2outputdataset/detector_radford.py](./gpt2outputdataset/detector_radford.py)
- Zero-shot method in [Mitchell et al. 2023](https://arxiv.org/abs/2301.11305v1) with suggestions in [Mireshghallah et al. 2023](https://arxiv.org/abs/2305.09859): [detector_detectgpt.py](./detector_detectgpt.py)