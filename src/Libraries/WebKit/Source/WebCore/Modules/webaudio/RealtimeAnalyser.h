/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include "AudioArray.h"
#include "AudioBus.h"
#include "NoiseInjectionPolicy.h"
#include <JavaScriptCore/Forward.h>
#include <memory>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FFTFrame;

class RealtimeAnalyser {
    WTF_MAKE_TZONE_ALLOCATED(RealtimeAnalyser);
    WTF_MAKE_NONCOPYABLE(RealtimeAnalyser);
public:
    explicit RealtimeAnalyser(OptionSet<NoiseInjectionPolicy>);
    virtual ~RealtimeAnalyser();

    size_t fftSize() const { return m_fftSize; }
    bool setFftSize(size_t);

    unsigned frequencyBinCount() const { return m_fftSize / 2; }

    void setMinDecibels(double k) { m_minDecibels = k; }
    double minDecibels() const { return m_minDecibels; }

    void setMaxDecibels(double k) { m_maxDecibels = k; }
    double maxDecibels() const { return m_maxDecibels; }

    void setSmoothingTimeConstant(double k) { m_smoothingTimeConstant = k; }
    double smoothingTimeConstant() const { return m_smoothingTimeConstant; }

    void getFloatFrequencyData(JSC::Float32Array&);
    void getByteFrequencyData(JSC::Uint8Array&);
    void getFloatTimeDomainData(JSC::Float32Array&);
    void getByteTimeDomainData(JSC::Uint8Array&);

    // The audio thread writes input data here.
    void writeInput(AudioBus*, size_t framesToProcess);

    static constexpr double DefaultSmoothingTimeConstant { 0.8 };
    static constexpr double DefaultMinDecibels { -100 };
    static constexpr double DefaultMaxDecibels { -30 };

    // All FFT implementations are expected to handle power-of-two sizes MinFFTSize <= size <= MaxFFTSize.
    static constexpr unsigned DefaultFFTSize { 2048 };
    static constexpr unsigned MinFFTSize { 32 };
    static constexpr unsigned MaxFFTSize { 32768 };
    static constexpr unsigned InputBufferSize { MaxFFTSize * 2 };

private:
    // The audio thread writes the input audio here.
    AudioFloatArray m_inputBuffer;
    unsigned m_writeIndex { 0 };

    // AudioBus used for downmixing input audio before copying it to m_inputBuffer.
    RefPtr<AudioBus> m_downmixBus;
    
    size_t m_fftSize { DefaultFFTSize };
    std::unique_ptr<FFTFrame> m_analysisFrame;
    void doFFTAnalysisIfNecessary();
    
    // doFFTAnalysisIfNecessary() stores the floating-point magnitude analysis data here.
    AudioFloatArray m_magnitudeBuffer { DefaultFFTSize / 2 };
    AudioFloatArray& magnitudeBuffer() { return m_magnitudeBuffer; }

    // A value between 0 and 1 which averages the previous version of m_magnitudeBuffer with the current analysis magnitude data.
    double m_smoothingTimeConstant { DefaultSmoothingTimeConstant };

    // The range used when converting when using getByteFrequencyData(). 
    double m_minDecibels { DefaultMinDecibels };
    double m_maxDecibels { DefaultMaxDecibels };

    // We should only do the FFT analysis once per render quantum.
    bool m_shouldDoFFTAnalysis { true };
    OptionSet<NoiseInjectionPolicy> m_noiseInjectionPolicies;
};

} // namespace WebCore
