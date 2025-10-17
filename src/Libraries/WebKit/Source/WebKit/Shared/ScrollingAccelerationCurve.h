/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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

#if ENABLE(MOMENTUM_EVENT_DISPATCHER)

#include <WebCore/FloatSize.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class NativeWebWheelEvent;

class ScrollingAccelerationCurve {
public:
    ScrollingAccelerationCurve(float gainLinear, float gainParabolic, float gainCubic, float gainQuartic, float tangentSpeedLinear, float tangentSpeedParabolicRoot, float resolution, float frameRate);

    static std::optional<ScrollingAccelerationCurve> fromNativeWheelEvent(const NativeWebWheelEvent&);

    static ScrollingAccelerationCurve interpolate(const ScrollingAccelerationCurve& from, const ScrollingAccelerationCurve& to, float amount);

    float gainLinear() const { return m_parameters.gainLinear; }
    float gainParabolic() const { return m_parameters.gainParabolic; }
    float gainCubic() const { return m_parameters.gainCubic; }
    float gainQuartic() const { return m_parameters.gainQuartic; }
    float tangentSpeedLinear() const { return m_parameters.tangentSpeedLinear; };
    float tangentSpeedParabolicRoot() const { return m_parameters.tangentSpeedParabolicRoot; };
    float resolution() const { return m_parameters.resolution; }
    float frameRate() const { return m_parameters.frameRate; }

    float accelerationFactor(float);

    friend bool operator==(const ScrollingAccelerationCurve& a, const ScrollingAccelerationCurve& b)
    {
        // Does not check m_intermediates.
        return a.m_parameters == b.m_parameters;
    }
    
private:
    friend TextStream& operator<<(TextStream&, const ScrollingAccelerationCurve&);

    void computeIntermediateValuesIfNeeded();

    float evaluateQuartic(float) const;

    struct Parameters {
        float gainLinear { 0 };
        float gainParabolic { 0 };
        float gainCubic { 0 };
        float gainQuartic { 0 };
        float tangentSpeedLinear { 0 };
        float tangentSpeedParabolicRoot { 0 };

        // FIXME: Resolution and frame rate are not technically properties
        // of the curve, just required to use it; they should be plumbed separately.
        float resolution { 0 };
        float frameRate { 0 };

        friend bool operator==(const Parameters&, const Parameters&) = default;
    };
    Parameters m_parameters;

    struct ComputedIntermediateValues {
        float tangentStartX { 0 };
        float tangentSlope { 0 };
        float tangentIntercept { 0 };
        float falloffStartX { 0 };
        float falloffSlope { 0 };
        float falloffIntercept { 0 };
    };

    std::optional<ComputedIntermediateValues> m_intermediates;
};

TextStream& operator<<(TextStream&, const ScrollingAccelerationCurve&);

} // namespace WebKit

#endif // ENABLE(MOMENTUM_EVENT_DISPATCHER)
