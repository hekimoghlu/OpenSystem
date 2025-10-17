/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#include "IIRDSPKernel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IIRDSPKernel);

IIRDSPKernel::IIRDSPKernel(IIRProcessor& processor)
    : AudioDSPKernel(&processor)
    , m_iirFilter(processor.feedforward(), processor.feedback())
    , m_tailTime(m_iirFilter.tailTime(processor.sampleRate(), processor.isFilterStable()))
{
}

void IIRDSPKernel::getFrequencyResponse(unsigned length, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse)
{
    ASSERT(frequencyHz.data());
    ASSERT(magResponse.data());
    ASSERT(phaseResponse.data());

    Vector<float> frequency(length);
    double nyquist = this->nyquist();

    // Convert from frequency in Hz to normalized frequency (0 -> 1), with 1 equal to the Nyquist frequency.
    for (unsigned k = 0; k < length; ++k)
        frequency[k] = frequencyHz[k] / nyquist;

    m_iirFilter.getFrequencyResponse(length, frequency.span(), magResponse, phaseResponse);
}

void IIRDSPKernel::process(std::span<const float> source, std::span<float> destination)
{
    ASSERT(source.data());
    ASSERT(destination.data());

    m_iirFilter.process(source, destination);
}

void IIRDSPKernel::reset()
{
    m_iirFilter.reset();
}

bool IIRDSPKernel::requiresTailProcessing() const
{
    // Always return true even if the tail time and latency might both be zero.
    return true;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
