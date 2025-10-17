/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "ExceptionOr.h"
#include "PeriodicWaveOptions.h"
#include <JavaScriptCore/Forward.h>
#include <memory>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class BaseAudioContext;

enum class ShouldDisableNormalization : bool { No, Yes };

class PeriodicWave final : public RefCounted<PeriodicWave> {
public:
    static Ref<PeriodicWave> createSine(float sampleRate);
    static Ref<PeriodicWave> createSquare(float sampleRate);
    static Ref<PeriodicWave> createSawtooth(float sampleRate);
    static Ref<PeriodicWave> createTriangle(float sampleRate);

    // Creates an arbitrary wave given the frequency components (Fourier coefficients).
    // FIXME: Remove once old constructor is phased out
    static Ref<PeriodicWave> create(float sampleRate, Float32Array& real, Float32Array& imag);
    
    static ExceptionOr<Ref<PeriodicWave>> create(BaseAudioContext&, PeriodicWaveOptions&& = { });

    // Returns pointers to the lower and higher wave data for the pitch range containing
    // the given fundamental frequency. These two tables are in adjacent "pitch" ranges
    // where the higher table will have the maximum number of partials which won't alias when played back
    // at this fundamental frequency. The lower wave is the next range containing fewer partials than the higher wave.
    // Interpolation between these two tables can be made according to tableInterpolationFactor.
    // Where values from 0 -> 1 interpolate between lower -> higher.
    void waveDataForFundamentalFrequency(float, std::span<float>& lowerWaveData, std::span<float>& higherWaveData, float& tableInterpolationFactor);

    // Returns the scalar multiplier to the oscillator frequency to calculate wave table phase increment.
    float rateScale() const { return m_rateScale; }

    unsigned periodicWaveSize() const;
    float sampleRate() const { return m_sampleRate; }

private:
    enum class Type {
        Sine,
        Square,
        Sawtooth,
        Triangle,
    };

    explicit PeriodicWave(float sampleRate);

    void generateBasicWaveform(Type);

    float m_sampleRate;
    unsigned m_numberOfRanges;

    // The lowest frequency (in Hertz) where playback will include all of the partials.
    // Playing back lower than this frequency will gradually lose more high-frequency information.
    // This frequency is quite low (~10Hz @ 44.1KHz)
    float m_lowestFundamentalFrequency;

    float m_rateScale;

    unsigned numberOfRanges() const { return m_numberOfRanges; }

    // Maximum possible number of partials (before culling).
    unsigned maxNumberOfPartials() const;

    unsigned numberOfPartialsForRange(unsigned rangeIndex) const;

    // Creates tables based on numberOfComponents Fourier coefficients.
    void createBandLimitedTables(std::span<const float> real, std::span<const float> imag, ShouldDisableNormalization = ShouldDisableNormalization::No);
    Vector<std::unique_ptr<AudioFloatArray>> m_bandLimitedTables;
};

} // namespace WebCore
