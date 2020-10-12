# FOCA: Feature-extractor Optimization through Classifier Anonymizatino [1]

Code of FOCA: Feature-extractor Optimization through Classifier Anonymization [1].  This source code reproduces the results (to be appeared [2]) using Wide ResNet [3] and PyramidNet [4] trained on CIFAR-10 and CIFAR-100 [5].

## Requirements

- computer running Linux
- NVIDIA GPU and NCCL
- Python version 3.6
- PyTorch

## Usage

Use python train.py to train a non-end-to-end model by FOCA. Here are some examples setting:

train cifar10 with Wide-ResNet and single gpu.  (default setting: --dataset='cifar10' --network='wide-resnet' --idGPU=[0])
> $python3 train.py 

## Author

Guoqing Liu and Ikuro Sato, Denso IT Laboratory, Inc.

## Reference

[1] Ikuro Sato, Kohta Ishikawa, Guoqing Liu, and Masayuki Tanaka, "Breaking Inter-Layer Co-Adaptation by Classifier Anonymization", Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

[2] To be appeared.

[3] Sergey Zagoruyko and Nikos Komodakis, "Wide Residual Networks", Proceedings of the British Machine Vision Conference (BMVC), 2016.

[4] Dongyoon Han, Jiwhan Kim, and Junmo Kim, "Deep Pyramidal Residual Networks", Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[5] Alex Krizhevskyf and Geoffrey Hinton, "Learning multiple layers of features from tiny images", Technical Report, University of Toronto, 2009.

## LICENSE

Copyright (C) 2020 Denso IT Laboratory, Inc.
All Rights Reserved

Denso IT Laboratory, Inc. retains sole and exclusive ownership of all
intellectual property rights including copyrights and patents related to this
Software.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the Software and accompanying documentation to use, copy, modify, merge,
publish, or distribute the Software or software derived from it for
non-commercial purposes, such as academic study, education and personal use,
subject to the following conditions:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
