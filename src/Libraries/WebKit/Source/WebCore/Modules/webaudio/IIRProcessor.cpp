/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include "IIRProcessor.h"

#include "IIRDSPKernel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IIRProcessor);

IIRProcessor::IIRProcessor(float sampleRate, unsigned numberOfChannels, const Vector<double>& feedforward, const Vector<double>& feedback, bool isFilterStable)
    : AudioDSPKernelProcessor(sampleRate, numberOfChannels)
    , m_feedforward(feedforward)
    , m_feedback(feedback)
    , m_isFilterStable(isFilterStable)
{
    unsigned feedbackLength = feedback.size();
    unsigned feedforwardLength = feedforward.size();
    ASSERT(feedbackLength > 0);
    ASSERT(feedforwardLength > 0);

    // Need to scale the feedback and feedforward coefficients appropriately.
    // (It's up to the caller to ensure feedbackCoef[0] is not 0.)
    ASSERT(!!feedback[0]);

    if (feedback[0] != 1) {
        // The provided filter is:
        //
        //   a[0]*y(n) + a[1]*y(n-1) + ... = b[0]*x(n) + b[1]*x(n-1) + ...
        //
        // We want the leading coefficient of y(n) to be 1:
        //
        //   y(n) + a[1]/a[0]*y(n-1) + ... = b[0]/a[0]*x(n) + b[1]/a[0]*x(n-1) + ...
        //
        // Thus, the feedback and feedforward coefficients need to be scaled by 1/a[0].
        float scale = feedback[0];
        for (unsigned k = 1; k < feedbackLength; ++k)
            m_feedback[k] /= scale;

        for (unsigned k = 0; k < feedforwardLength; ++k)
            m_feedforward[k] /= scale;

        // The IIRFilter checks to make sure this coefficient is 1, so make it so.
        m_feedback[0] = 1;
    }

    m_responseKernel = makeUnique<IIRDSPKernel>(*this);
}

IIRProcessor::~IIRProcessor()
{
    if (isInitialized())
        uninitialize();
}

std::unique_ptr<AudioDSPKernel> IIRProcessor::createKernel()
{
    return makeUnique<IIRDSPKernel>(*this);
}

void IIRProcessor::process(const AudioBus* source, AudioBus* destination, size_t framesToProcess)
{
    if (!isInitialized()) {
        destination->zero();
        return;
    }

    // For each channel of our input, process using the corresponding IIRDSPKernel
    // into the output channel.
    for (size_t i = 0; i < m_kernels.size(); ++i)
        m_kernels[i]->process(source->channel(i)->span().first(framesToProcess), destination->channel(i)->mutableSpan());
}

void IIRProcessor::getFrequencyResponse(unsigned length, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse)
{
    m_responseKernel->getFrequencyResponse(length, frequencyHz, magResponse, phaseResponse);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
