/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "config.h"

#if ENABLE(WEB_AUDIO)

#include "UpSampler.h"

#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UpSampler);

UpSampler::UpSampler(size_t inputBlockSize)
    : m_inputBlockSize(inputBlockSize)
    , m_kernel(DefaultKernelSize)
    , m_convolver(inputBlockSize)
    , m_tempBuffer(inputBlockSize)
    , m_inputBuffer(inputBlockSize * 2)
{
    initializeKernel();
}

void UpSampler::initializeKernel()
{
    // Blackman window parameters.
    double alpha = 0.16;
    double a0 = 0.5 * (1.0 - alpha);
    double a1 = 0.5;
    double a2 = 0.5 * alpha;

    int n = m_kernel.size();
    int halfSize = n / 2;
    double subsampleOffset = -0.5;

    for (int i = 0; i < n; ++i) {
        // Compute the sinc() with offset.
        double s = piDouble * (i - halfSize - subsampleOffset);
        double sinc = !s ? 1.0 : sin(s) / s;

        // Compute Blackman window, matching the offset of the sinc().
        double x = (i - subsampleOffset) / n;
        double window = a0 - a1 * cos(2.0 * piDouble * x) + a2 * cos(4.0 * piDouble * x);

        // Window the sinc() function.
        m_kernel[i] = sinc * window;
    }
}

void UpSampler::process(std::span<const float> source, std::span<float> destination)
{
    bool isInputBlockSizeGood = source.size() == m_inputBlockSize;
    ASSERT(isInputBlockSizeGood);
    if (!isInputBlockSizeGood)
        return;

    bool isTempBufferGood = source.size() == m_tempBuffer.size();
    ASSERT(isTempBufferGood);
    if (!isTempBufferGood)
        return;

    bool isKernelGood = m_kernel.size() == DefaultKernelSize;
    ASSERT(isKernelGood);
    if (!isKernelGood)
        return;

    size_t halfSize = m_kernel.size() / 2;

    // Copy source samples to 2nd half of input buffer.
    bool isInputBufferGood = m_inputBuffer.size() == source.size() * 2 && halfSize <= source.size();
    ASSERT(isInputBufferGood);
    if (!isInputBufferGood)
        return;

    auto inputBuffer = m_inputBuffer.span();
    auto inputP = inputBuffer.subspan(source.size());
    memcpySpan(inputP, source);

    // Copy even sample-frames 0,2,4,6... (delayed by the linear phase delay) directly into destination.
    auto inputPMinusHalfSize = inputBuffer.subspan(inputP.data() - inputBuffer.data() - halfSize);
    for (size_t i = 0; i < source.size(); ++i)
        destination[i * 2] = inputPMinusHalfSize[i];

    // Compute odd sample-frames 1,3,5,7...
    auto oddSamplesP = m_tempBuffer.span();
    m_convolver.process(&m_kernel, source, oddSamplesP);

    for (size_t i = 0; i < source.size(); ++i)
        destination[i * 2 + 1] = oddSamplesP[i];

    // Copy 2nd half of input buffer to 1st half.
    memcpySpan(m_inputBuffer.span(), inputP);
}

void UpSampler::reset()
{
    m_convolver.reset();
    m_inputBuffer.zero();
}

size_t UpSampler::latencyFrames() const
{
    // Divide by two since this is a linear phase kernel and the delay is at the center of the kernel.
    return m_kernel.size() / 2;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
