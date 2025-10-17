/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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

#include "DownSampler.h"

#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DownSampler);

DownSampler::DownSampler(size_t inputBlockSize)
    : m_inputBlockSize(inputBlockSize)
    , m_convolver(inputBlockSize / 2) // runs at 1/2 source sample-rate
    , m_tempBuffer(inputBlockSize / 2)
    , m_inputBuffer(inputBlockSize * 2)
{
    initializeKernel();
}

void DownSampler::initializeKernel()
{
    // Blackman window parameters.
    double alpha = 0.16;
    double a0 = 0.5 * (1.0 - alpha);
    double a1 = 0.5;
    double a2 = 0.5 * alpha;

    int n = DefaultKernelSize;
    int halfSize = n / 2;

    // Half-band filter.
    double sincScaleFactor = 0.5;

    // Compute only the odd terms because the even ones are zero, except
    // right in the middle at halfSize, which is 0.5 and we'll handle specially during processing
    // after doing the main convolution using m_reducedKernel.
    for (int i = 1; i < n; i += 2) {
        // Compute the sinc() with offset.
        double s = sincScaleFactor * piDouble * (i - halfSize);
        double sinc = !s ? 1.0 : sin(s) / s;
        sinc *= sincScaleFactor;

        // Compute Blackman window, matching the offset of the sinc().
        double x = static_cast<double>(i) / n;
        double window = a0 - a1 * cos(2.0 * piDouble * x) + a2 * cos(4.0 * piDouble * x);

        // Window the sinc() function.
        // Then store only the odd terms in the kernel.
        // In a sense, this is shifting forward in time by one sample-frame at the destination sample-rate.
        m_reducedKernel[(i - 1) / 2] = sinc * window;
    }
}

void DownSampler::process(std::span<const float> source, std::span<float> destination)
{
    bool isInputBlockSizeGood = source.size() == m_inputBlockSize;
    ASSERT(isInputBlockSizeGood);
    if (!isInputBlockSizeGood)
        return;

    size_t destFramesToProcess = source.size() / 2;

    bool isTempBufferGood = destFramesToProcess == m_tempBuffer.size();
    ASSERT(isTempBufferGood);
    if (!isTempBufferGood)
        return;

    bool isReducedKernelGood = m_reducedKernel.size() == DefaultKernelSize / 2;
    ASSERT(isReducedKernelGood);
    if (!isReducedKernelGood)
        return;

    size_t halfSize = DefaultKernelSize / 2;

    // Copy source samples to 2nd half of input buffer.
    bool isInputBufferGood = m_inputBuffer.size() == source.size() * 2 && halfSize <= source.size();
    ASSERT(isInputBufferGood);
    if (!isInputBufferGood)
        return;

    auto inputBuffer = m_inputBuffer.span();
    auto inputP = inputBuffer.subspan(source.size());
    memcpySpan(inputP, source);

    // Copy the odd sample-frames from source, delayed by one sample-frame (destination sample-rate)
    // to match shifting forward in time in m_reducedKernel.
    auto oddSamplesP = m_tempBuffer.span().first(destFramesToProcess);
    auto inputPMinusOne = inputBuffer.subspan(inputP.data() - inputBuffer.data() - 1);
    for (size_t i = 0; i < oddSamplesP.size(); ++i)
        oddSamplesP[i] = inputPMinusOne[i * 2];

    // Actually process oddSamplesP with m_reducedKernel for efficiency.
    // The theoretical kernel is double this size with 0 values for even terms (except center).
    m_convolver.process(&m_reducedKernel, oddSamplesP, destination);

    // Now, account for the 0.5 term right in the middle of the kernel.
    // This amounts to a delay-line of length halfSize (at the source sample-rate),
    // scaled by 0.5.

    // Sum into the destination.
    auto inputPMinusHalfSize = inputBuffer.subspan(inputP.data() - inputBuffer.data() - halfSize);
    for (size_t i = 0; i < destFramesToProcess; ++i)
        destination[i] += 0.5 * inputPMinusHalfSize[i * 2];

    // Copy 2nd half of input buffer to 1st half.
    memcpySpan(m_inputBuffer.span(), inputP);
}

void DownSampler::reset()
{
    m_convolver.reset();
    m_inputBuffer.zero();
}

size_t DownSampler::latencyFrames() const
{
    // Divide by two since this is a linear phase kernel and the delay is at the center of the kernel.
    return m_reducedKernel.size() / 2;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
