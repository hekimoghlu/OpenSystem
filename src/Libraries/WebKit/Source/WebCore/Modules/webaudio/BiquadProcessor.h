/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

#include "AudioDSPKernel.h"
#include "AudioDSPKernelProcessor.h"
#include "AudioNode.h"
#include "AudioParam.h"
#include "Biquad.h"
#include "BiquadFilterType.h"
#include <memory>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// BiquadProcessor is an AudioDSPKernelProcessor which uses Biquad objects to implement several common filters.

class BiquadProcessor final : public AudioDSPKernelProcessor {
    WTF_MAKE_TZONE_ALLOCATED(BiquadProcessor);
public:
    BiquadProcessor(BaseAudioContext&, float sampleRate, size_t numberOfChannels, bool autoInitialize);

    virtual ~BiquadProcessor();
    
    std::unique_ptr<AudioDSPKernel> createKernel() override;
        
    void process(const AudioBus* source, AudioBus* destination, size_t framesToProcess) override;
    void processOnlyAudioParams(size_t framesToProcess) final;

    // Get the magnitude and phase response of the filter at the given
    // set of frequencies (in Hz). The phase response is in radians.
    void getFrequencyResponse(unsigned nFrequencies, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse);

    void checkForDirtyCoefficients();
    
    bool filterCoefficientsDirty() const { return m_filterCoefficientsDirty; }
    bool hasSampleAccurateValues() const { return m_hasSampleAccurateValues; }

    AudioParam& parameter1() { return m_parameter1.get(); }
    AudioParam& parameter2() { return m_parameter2.get(); }
    AudioParam& parameter3() { return m_parameter3.get(); }
    AudioParam& parameter4() { return m_parameter4.get(); }

    BiquadFilterType type() const { return m_type; }
    void setType(BiquadFilterType);

    bool shouldUseARate() const { return m_shouldUseARate; }

private:
    Type processorType() const final { return Type::Biquad; }

    BiquadFilterType m_type { BiquadFilterType::Lowpass };

    Ref<AudioParam> m_parameter1;
    Ref<AudioParam> m_parameter2;
    Ref<AudioParam> m_parameter3;
    Ref<AudioParam> m_parameter4;

    // so DSP kernels know when to re-compute coefficients
    bool m_filterCoefficientsDirty { true };

    // Set to true if any of the filter parameters are sample-accurate.
    bool m_hasSampleAccurateValues { false };

    bool m_shouldUseARate { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::BiquadProcessor) \
    static bool isType(const WebCore::AudioProcessor& processor) { return processor.processorType() == WebCore::AudioProcessor::Type::Biquad; } \
SPECIALIZE_TYPE_TRAITS_END()
