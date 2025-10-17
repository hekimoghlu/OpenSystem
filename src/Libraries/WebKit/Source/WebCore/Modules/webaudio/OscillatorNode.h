/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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

#include "AudioScheduledSourceNode.h"
#include "OscillatorOptions.h"
#include "OscillatorType.h"
#include "PeriodicWave.h"
#include <wtf/Lock.h>

namespace WebCore {

// OscillatorNode is an audio generator of periodic waveforms.

class OscillatorNode final : public AudioScheduledSourceNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OscillatorNode);
public:
    static ExceptionOr<Ref<OscillatorNode>> create(BaseAudioContext&, const OscillatorOptions& = { });

    virtual ~OscillatorNode();

    OscillatorType typeForBindings() const { ASSERT(isMainThread()); return m_type; }
    ExceptionOr<void> setTypeForBindings(OscillatorType);

    AudioParam* frequency() { return m_frequency.get(); }
    AudioParam* detune() { return m_detune.get(); }

    void setPeriodicWave(PeriodicWave&);

private:
    OscillatorNode(BaseAudioContext&, const OscillatorOptions& = { });

    void process(size_t framesToProcess) final;

    double tailTime() const final { return 0; }
    double latencyTime() const final { return 0; }

    // Returns true if there are sample-accurate timeline parameter changes.
    bool calculateSampleAccuratePhaseIncrements(size_t framesToProcess) WTF_REQUIRES_LOCK(m_processLock);

    double processARate(int, std::span<float> destP, double virtualReadIndex, std::span<float> phaseIncrements) WTF_REQUIRES_LOCK(m_processLock);
    double processKRate(int, std::span<float> destP, double virtualReadIndex) WTF_REQUIRES_LOCK(m_processLock);

    bool propagatesSilence() const final;

    float noiseInjectionMultiplier() const final { return 0.01; }

    // One of the waveform types defined in the enum.
    OscillatorType m_type; // Only used on the main thread.
    
    // Frequency value in Hertz.
    RefPtr<AudioParam> m_frequency;

    // Detune value (deviating from the frequency) in Cents.
    RefPtr<AudioParam> m_detune;

    bool m_firstRender { true };

    // m_virtualReadIndex is a sample-frame index into our buffer representing the current playback position.
    // Since it's floating-point, it has sub-sample accuracy.
    double m_virtualReadIndex { 0 };

    // This synchronizes process().
    mutable Lock m_processLock;

    // Stores sample-accurate values calculated according to frequency and detune.
    AudioFloatArray m_phaseIncrements;
    AudioFloatArray m_detuneValues;
    
    RefPtr<PeriodicWave> m_periodicWave WTF_GUARDED_BY_LOCK(m_processLock);
};

String convertEnumerationToString(OscillatorType); // In JSOscillatorNode.cpp

} // namespace WebCore

namespace WTF {

template<> struct LogArgument<WebCore::OscillatorType> {
    static String toString(WebCore::OscillatorType type) { return convertEnumerationToString(type); }
};

} // namespace WTF
