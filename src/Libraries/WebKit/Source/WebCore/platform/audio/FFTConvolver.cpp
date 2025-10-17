/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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

#include "FFTConvolver.h"

#include "VectorMath.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {
    
WTF_MAKE_TZONE_ALLOCATED_IMPL(FFTConvolver);

FFTConvolver::FFTConvolver(size_t fftSize)
    : m_frame(fftSize)
    , m_inputBuffer(fftSize) // 2nd half of buffer is always zeroed
    , m_outputBuffer(fftSize)
    , m_lastOverlapBuffer(fftSize / 2)
{
}

void FFTConvolver::process(FFTFrame* fftKernel, std::span<const float> source, std::span<float> destination)
{
    size_t halfSize = fftSize() / 2;

    // source.size() must be an exact multiple of halfSize,
    // or halfSize is a multiple of source.size() when halfSize > source.size().
    bool isGood = !(halfSize % source.size() && source.size() % halfSize);
    ASSERT(isGood);
    if (!isGood)
        return;

    size_t numberOfDivisions = halfSize <= source.size() ? (source.size() / halfSize) : 1;
    size_t divisionSize = numberOfDivisions == 1 ? source.size() : halfSize;

    for (size_t i = 0; i < numberOfDivisions; ++i, skip(source, divisionSize), skip(destination, divisionSize)) {
        // Copy samples to input buffer (note contraint above!)
        auto inputP = m_inputBuffer.span();

        // Sanity check
        bool isCopyGood1 = source.data() && inputP.data() && m_readWriteIndex + divisionSize <= m_inputBuffer.size();
        ASSERT(isCopyGood1);
        if (!isCopyGood1)
            return;

        IGNORE_WARNINGS_BEGIN("restrict")
        memcpySpan(inputP.subspan(m_readWriteIndex), source.first(divisionSize));
        IGNORE_WARNINGS_END

        // Copy samples from output buffer
        auto outputP = m_outputBuffer.span();

        // Sanity check
        bool isCopyGood2 = destination.data() && outputP.data() && m_readWriteIndex + divisionSize <= m_outputBuffer.size();
        ASSERT(isCopyGood2);
        if (!isCopyGood2)
            return;

        memcpySpan(destination, outputP.subspan(m_readWriteIndex, divisionSize));
        m_readWriteIndex += divisionSize;

        // Check if it's time to perform the next FFT
        if (m_readWriteIndex == halfSize) {
            // The input buffer is now filled (get frequency-domain version)
            m_frame.doFFT(m_inputBuffer.span());
            m_frame.multiply(*fftKernel);
            m_frame.doInverseFFT(m_outputBuffer.span());

            // Overlap-add 1st half from previous time
            VectorMath::add(m_outputBuffer.span().first(halfSize), m_lastOverlapBuffer.span().first(halfSize), m_outputBuffer.span());

            // Finally, save 2nd half of result
            bool isCopyGood3 = m_outputBuffer.size() == 2 * halfSize && m_lastOverlapBuffer.size() == halfSize;
            ASSERT(isCopyGood3);
            if (!isCopyGood3)
                return;

            memcpySpan(m_lastOverlapBuffer.span(), m_outputBuffer.span().subspan(halfSize, halfSize));

            // Reset index back to start for next time
            m_readWriteIndex = 0;
        }
    }
}

void FFTConvolver::reset()
{
    m_lastOverlapBuffer.zero();
    m_readWriteIndex = 0;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
