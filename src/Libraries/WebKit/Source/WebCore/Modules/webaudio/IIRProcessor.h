/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#pragma once

#include "AudioDSPKernelProcessor.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class IIRDSPKernel;

class IIRProcessor final : public AudioDSPKernelProcessor {
    WTF_MAKE_TZONE_ALLOCATED(IIRProcessor);
public:
    IIRProcessor(float sampleRate, unsigned numberOfChannels, const Vector<double>& feedforward, const Vector<double>& feedback, bool isFilterStable);
    ~IIRProcessor();

    // AudioDSPKernelProcessor.
    void process(const AudioBus* source, AudioBus* destination, size_t framesToProcess) final;
    std::unique_ptr<AudioDSPKernel> createKernel() final;

    // Get the magnitude and phase response of the filter at the given set of frequencies (in Hz). The phase response is in radians.
    void getFrequencyResponse(unsigned length, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse);

    const Vector<double>& feedforward() const { return m_feedforward; }
    const Vector<double>& feedback() const { return m_feedback; }
    bool isFilterStable() const { return m_isFilterStable; }

private:
    Type processorType() const final { return Type::IIR; }

    Vector<double> m_feedforward;
    Vector<double> m_feedback;
    bool m_isFilterStable;

    std::unique_ptr<IIRDSPKernel> m_responseKernel;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::IIRProcessor) \
    static bool isType(const WebCore::AudioProcessor& processor) { return processor.processorType() == WebCore::AudioProcessor::Type::IIR; } \
SPECIALIZE_TYPE_TRAITS_END()
