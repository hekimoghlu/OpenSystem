/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef UpSampler_h
#define UpSampler_h

#include "AudioArray.h"
#include "DirectConvolver.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// UpSampler up-samples the source stream by a factor of 2x.

class UpSampler final {
    WTF_MAKE_TZONE_ALLOCATED(UpSampler);
public:
    explicit UpSampler(size_t inputBlockSize);

    // The destination buffer |destination| is of size source.size() * 2.
    void process(std::span<const float> source, std::span<float> destination);

    void reset();

    // Latency based on the source sample-rate.
    size_t latencyFrames() const;

private:
    enum { DefaultKernelSize = 128 };

    size_t m_inputBlockSize;

    // Computes ideal band-limited filter coefficients to sample in between each source sample-frame.
    // This filter will be used to compute the odd sample-frames of the output.
    void initializeKernel();
    AudioFloatArray m_kernel;

    // Computes the odd sample-frames of the output.
    DirectConvolver m_convolver;

    AudioFloatArray m_tempBuffer;

    // Delay line for generating the even sample-frames of the output.
    // The source samples are delayed exactly to match the linear phase delay of the FIR filter (convolution)
    // used to generate the odd sample-frames of the output.
    AudioFloatArray m_inputBuffer;
};

} // namespace WebCore

#endif // UpSampler_h
