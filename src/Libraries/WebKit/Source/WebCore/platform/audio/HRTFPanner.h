/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#ifndef HRTFPanner_h
#define HRTFPanner_h

#include "DelayDSPKernel.h"
#include "FFTConvolver.h"
#include "Panner.h"

namespace WebCore {

class HRTFDatabaseLoader;

class HRTFPanner final : public Panner {
public:
    explicit HRTFPanner(float sampleRate, HRTFDatabaseLoader*);
    virtual ~HRTFPanner();

    // Panner
    void pan(double azimuth, double elevation, const AudioBus* inputBus, AudioBus* outputBus, size_t framesToProcess) final;
    void panWithSampleAccurateValues(std::span<double> azimuth, std::span<double> elevation, const AudioBus* inputBus, AudioBus* outputBus, size_t framesToProcess) final;
    void reset() override;

    size_t fftSize() const { return fftSizeForSampleRate(m_sampleRate); }
    static size_t fftSizeForSampleRate(float sampleRate);

    float sampleRate() const { return m_sampleRate; }

    double tailTime() const override;
    double latencyTime() const override;
    bool requiresTailProcessing() const final;

private:
    // Given an azimuth angle in the range -180 -> +180, returns the corresponding azimuth index for the database,
    // and azimuthBlend which is an interpolation value from 0 -> 1.
    int calculateDesiredAzimuthIndexAndBlend(double azimuth, double& azimuthBlend);

    RefPtr<HRTFDatabaseLoader> m_databaseLoader;

    float m_sampleRate;

    // We maintain two sets of convolvers for smooth cross-faded interpolations when
    // then azimuth and elevation are dynamically changing.
    // When the azimuth and elevation are not changing, we simply process with one of the two sets.
    // Initially we use CrossfadeSelection1 corresponding to m_convolverL1 and m_convolverR1.
    // Whenever the azimuth or elevation changes, a crossfade is initiated to transition
    // to the new position. So if we're currently processing with CrossfadeSelection1, then
    // we transition to CrossfadeSelection2 (and vice versa).
    // If we're in the middle of a transition, then we wait until it is complete before
    // initiating a new transition.

    // Selects either the convolver set (m_convolverL1, m_convolverR1) or (m_convolverL2, m_convolverR2).
    enum CrossfadeSelection {
        CrossfadeSelection1,
        CrossfadeSelection2
    };

    CrossfadeSelection m_crossfadeSelection { CrossfadeSelection1 };

    // azimuth/elevation for CrossfadeSelection1.
    std::optional<int> m_azimuthIndex1;
    double m_elevation1 { 0 };

    // azimuth/elevation for CrossfadeSelection2.
    std::optional<int> m_azimuthIndex2;
    double m_elevation2 { 0 };

    // A crossfade value 0 <= m_crossfadeX <= 1.
    float m_crossfadeX { 0 };

    // Per-sample-frame crossfade value increment.
    float m_crossfadeIncr { 0 };

    FFTConvolver m_convolverL1;
    FFTConvolver m_convolverR1;
    FFTConvolver m_convolverL2;
    FFTConvolver m_convolverR2;

    DelayDSPKernel m_delayLineL;
    DelayDSPKernel m_delayLineR;

    AudioFloatArray m_tempL1;
    AudioFloatArray m_tempR1;
    AudioFloatArray m_tempL2;
    AudioFloatArray m_tempR2;
};

} // namespace WebCore

#endif // HRTFPanner_h
