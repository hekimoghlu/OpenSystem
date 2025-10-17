/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#ifndef Biquad_h
#define Biquad_h

#include "AudioArray.h"
#include <complex>
#include <sys/types.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// A basic biquad (two-zero / two-pole digital filter)
//
// It can be configured to a number of common and very useful filters:
//    lowpass, highpass, shelving, parameteric, notch, allpass, ...

class Biquad final {
    WTF_MAKE_TZONE_ALLOCATED(Biquad);
public:   
    Biquad();
    ~Biquad();

    void process(std::span<const float> source, std::span<float> destination);

    bool hasSampleAccurateValues() const { return m_hasSampleAccurateValues; }
    void setHasSampleAccurateValues(bool hasSampleAccurateValues) { m_hasSampleAccurateValues = hasSampleAccurateValues; }

    // frequency is 0 - 1 normalized, resonance and dbGain are in decibels.
    // Q is a unitless quality factor.
    void setLowpassParams(size_t index, double frequency, double resonance);
    void setHighpassParams(size_t index, double frequency, double resonance);
    void setBandpassParams(size_t index, double frequency, double Q);
    void setLowShelfParams(size_t index, double frequency, double dbGain);
    void setHighShelfParams(size_t index, double frequency, double dbGain);
    void setPeakingParams(size_t index, double frequency, double Q, double dbGain);
    void setAllpassParams(size_t index, double frequency, double Q);
    void setNotchParams(size_t index, double frequency, double Q);

    // Resets filter state
    void reset();

    // Filter response at a set of n frequencies. The magnitude and
    // phase response are returned in magResponse and phaseResponse.
    // The phase response is in radians.
    void getFrequencyResponse(unsigned nFrequencies, std::span<const float> frequency, std::span<float> magResponse, std::span<float> phaseResponse);

    // Compute tail frame based on the filter coefficents at index
    // |coefIndex|. The tail frame is the frame number where the
    // impulse response of the filter falls below a threshold value.
    // The maximum allowed frame value is given by |maxFrame|. This
    // limits how much work is done in computing the frame number.
    double tailFrame(size_t coefIndex, double maxFrame);

private:
    void setNormalizedCoefficients(size_t index, double b0, double b1, double b2, double a0, double a1, double a2);

    // Filter coefficients. The filter is defined as
    //
    // y[n] + m_a1*y[n-1] + m_a2*y[n-2] = m_b0*x[n] + m_b1*x[n-1] + m_b2*x[n-2].
    AudioDoubleArray m_b0;
    AudioDoubleArray m_b1;
    AudioDoubleArray m_b2;
    AudioDoubleArray m_a1;
    AudioDoubleArray m_a2;

#if USE(ACCELERATE)
    void processFast(std::span<const float> source, std::span<float> destination);
    void processSliceFast(std::span<double> source, std::span<double> destination, std::span<double> coefficients, size_t framesToProcess);
    AudioDoubleArray m_inputBuffer;
    AudioDoubleArray m_outputBuffer;
#endif

    // Filter memory
    double m_x1; // input delayed by 1 sample
    double m_x2; // input delayed by 2 samples
    double m_y1; // output delayed by 1 sample
    double m_y2; // output delayed by 2 samples

    bool m_hasSampleAccurateValues { false };
};

} // namespace WebCore

#endif // Biquad_h
