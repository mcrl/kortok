# README for TRG Researchers

## Meeting requirements for FAIRSeq and KorTok

As fairseq requirements have been changed, we need different version of packages.
We update package version to up-to-date ones, and maintain the code.

```bash
conda activate kakao-kortok
conda install python=3.9 -y # sentencepiece only supports python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install sentencepiece packaging
conda install scikit-learn

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
