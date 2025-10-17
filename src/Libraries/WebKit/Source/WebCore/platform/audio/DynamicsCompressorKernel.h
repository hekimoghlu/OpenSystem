/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#ifndef DynamicsCompressorKernel_h
#define DynamicsCompressorKernel_h

#include "AudioArray.h"
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class DynamicsCompressorKernel final {
    WTF_MAKE_TZONE_ALLOCATED(DynamicsCompressorKernel);
public:
    explicit DynamicsCompressorKernel(float sampleRate, unsigned numberOfChannels);

    void setNumberOfChannels(unsigned);

    // Performs stereo-linked compression.
    void process(std::span<std::span<const float>> sourceChannels,
                 std::span<std::span<float>> destinationChannels,
                 unsigned framesToProcess,
                 float dbThreshold,
                 float dbKnee,
                 float ratio,
                 float attackTime,
                 float releaseTime,
                 float preDelayTime,
                 float dbPostGain,
                 float effectBlend,

                 float releaseZone1,
                 float releaseZone2,
                 float releaseZone3,
                 float releaseZone4
                 );

    void reset();

    unsigned latencyFrames() const { return m_lastPreDelayFrames; }

    float sampleRate() const { return m_sampleRate; }

    float meteringGain() const { return m_meteringGain; }

    double tailTime() const;

protected:
    static constexpr float uninitializedValue = -1;
    float m_sampleRate;

    float m_detectorAverage;
    float m_compressorGain;

    // Metering
    float m_meteringReleaseK;
    float m_meteringGain;

    // Lookahead section.
    enum { MaxPreDelayFrames = 1024 };
    enum { MaxPreDelayFramesMask = MaxPreDelayFrames - 1 };
    enum { DefaultPreDelayFrames = 256 }; // setPreDelayTime() will override this initial value
    unsigned m_lastPreDelayFrames { DefaultPreDelayFrames };
    void setPreDelayTime(float);

    Vector<std::unique_ptr<AudioFloatArray>> m_preDelayBuffers;
    int m_preDelayReadIndex { 0 };
    int m_preDelayWriteIndex { DefaultPreDelayFrames };

    float m_maxAttackCompressionDiffDb;

    // Static compression curve.
    float kneeCurve(float x, float k);
    float saturate(float x, float k);
    float slopeAt(float x, float k);
    float kAtSlope(float desiredSlope);

    float updateStaticCurveParameters(float dbThreshold, float dbKnee, float ratio);

    // Amount of input change in dB required for 1 dB of output change.
    // This applies to the portion of the curve above m_kneeThresholdDb (see below).
    float m_ratio { uninitializedValue };
    float m_slope { uninitializedValue }; // Inverse ratio.

    // The input to output change below the threshold is linear 1:1.
    float m_linearThreshold { uninitializedValue };
    float m_dbThreshold { uninitializedValue };

    // m_dbKnee is the number of dB above the threshold before we enter the "ratio" portion of the curve.
    // m_kneeThresholdDb = m_dbThreshold + m_dbKnee
    // The portion between m_dbThreshold and m_kneeThresholdDb is the "soft knee" portion of the curve
    // which transitions smoothly from the linear portion to the ratio portion.
    float m_dbKnee { uninitializedValue };
    float m_kneeThreshold { uninitializedValue };
    float m_kneeThresholdDb { uninitializedValue };
    float m_ykneeThresholdDb { uninitializedValue };

    // Internal parameter for the knee portion of the curve.
    float m_K { uninitializedValue };
};

} // namespace WebCore

#endif // DynamicsCompressorKernel_h
