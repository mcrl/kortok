# README for TRG Researchers

## Meeting requirements for FAIRSeq and KorTok

As fairseq requirements have been changed, we need different version of packages.
We update package version to up-to-date ones, and maintain the code.

```bash
conda activate kakao-kortok
conda install python=3.9 -y # sentencepiece only supports python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install sentencepiece packaging konlpy scikit-learn mosestokenizer scipy transformers mecab-python3

pushd ~
git clone https://github.com/NVIDIA/apex
pushd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
popd
pip install pyarrow

git clone https://github.com/pytorch/fairseq
pushd fairseq
pip install --editable ./
popd
popd
```

## Meeting Requirements for Huggingface and KorTok

```bash
conda activate kortok-gpt
conda install python=3.9 -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# install jaxlib via pypi.
# As our cluster supported cudnn 8.7, we installed jaxlib 0.4.10+cuda11.cudnn86
pip install --upgrade "jax[cuda11_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax

# install other huggingface libraries
pip install datasets evaluate accelerate

# miscellaneous package
pip install pytest
```
