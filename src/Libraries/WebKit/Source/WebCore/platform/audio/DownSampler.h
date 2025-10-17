/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#ifndef DownSampler_h
#define DownSampler_h

#include "AudioArray.h"
#include "DirectConvolver.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// DownSampler down-samples the source stream by a factor of 2x.

class DownSampler final {
    WTF_MAKE_TZONE_ALLOCATED(DownSampler);
public:
    explicit DownSampler(size_t inputBlockSize);

    // The destination buffer |destination| is of size source.size() / 2.
    void process(std::span<const float> source, std::span<float> destination);

    void reset();

    // Latency based on the destination sample-rate.
    size_t latencyFrames() const;

private:
    enum { DefaultKernelSize = 256 };

    size_t m_inputBlockSize;

    // Computes ideal band-limited half-band filter coefficients.
    // In other words, filter out all frequencies higher than 0.25 * Nyquist.
    void initializeKernel();
    AudioFloatArray m_reducedKernel { DefaultKernelSize / 2 };

    // Half-band filter.
    DirectConvolver m_convolver;

    AudioFloatArray m_tempBuffer;

    // Used as delay-line (FIR filter history) for the input samples to account for the 0.5 term right in the middle of the kernel.
    AudioFloatArray m_inputBuffer;
};

} // namespace WebCore

#endif // DownSampler_h
