/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#include "ScrollingAccelerationCurve.h"

#if ENABLE(MOMENTUM_EVENT_DISPATCHER)

#include "ArgumentCoders.h"
#include <wtf/text/TextStream.h>

namespace WebKit {

ScrollingAccelerationCurve::ScrollingAccelerationCurve(float gainLinear, float gainParabolic, float gainCubic, float gainQuartic, float tangentSpeedLinear, float tangentSpeedParabolicRoot, float resolution, float frameRate)
    : m_parameters { gainLinear, gainParabolic, gainCubic, gainQuartic, tangentSpeedLinear, tangentSpeedParabolicRoot, resolution, frameRate }
{
}

ScrollingAccelerationCurve ScrollingAccelerationCurve::interpolate(const ScrollingAccelerationCurve& from, const ScrollingAccelerationCurve& to, float amount)
{
    auto interpolate = [&] (float a, float b) -> float {
        return a + amount * (b - a);
    };

    auto gainLinear = interpolate(from.m_parameters.gainLinear, to.m_parameters.gainLinear);
    auto gainParabolic = interpolate(from.m_parameters.gainParabolic, to.m_parameters.gainParabolic);
    auto gainCubic = interpolate(from.m_parameters.gainCubic, to.m_parameters.gainCubic);
    auto gainQuartic = interpolate(from.m_parameters.gainQuartic, to.m_parameters.gainQuartic);

    auto tangentSpeedLinear = interpolate(from.m_parameters.tangentSpeedLinear, to.m_parameters.tangentSpeedLinear);
    auto tangentSpeedParabolicRoot = interpolate(from.m_parameters.tangentSpeedParabolicRoot, to.m_parameters.tangentSpeedParabolicRoot);

    return { gainLinear, gainParabolic, gainCubic, gainQuartic, tangentSpeedLinear, tangentSpeedParabolicRoot, from.m_parameters.resolution, from.m_parameters.frameRate };
}

void ScrollingAccelerationCurve::computeIntermediateValuesIfNeeded()
{
    if (m_intermediates)
        return;
    m_intermediates = ComputedIntermediateValues { };

    float tangentInitialY;
    m_intermediates->tangentStartX = std::numeric_limits<float>::max();
    m_intermediates->falloffStartX = std::numeric_limits<float>::max();

    auto quarticSlope = [&](float x) {
        return (4 * std::pow(x, 3) * std::pow(m_parameters.gainQuartic, 4)) + (3 * std::pow(x, 2) * std::pow(m_parameters.gainCubic, 3)) + (2 * x * std::pow(m_parameters.gainParabolic, 2)) + m_parameters.gainLinear;
    };

    if (m_parameters.tangentSpeedLinear) {
        tangentInitialY = evaluateQuartic(m_parameters.tangentSpeedLinear);
        m_intermediates->tangentSlope = quarticSlope(m_parameters.tangentSpeedLinear);
        m_intermediates->tangentIntercept = tangentInitialY - m_intermediates->tangentSlope * m_parameters.tangentSpeedLinear;
        m_intermediates->tangentStartX = m_parameters.tangentSpeedLinear;

        if (m_parameters.tangentSpeedParabolicRoot) {
            float y1 = m_intermediates->tangentSlope * m_parameters.tangentSpeedParabolicRoot + m_intermediates->tangentIntercept;
            m_intermediates->falloffSlope = 2 * y1 * m_intermediates->tangentSlope;
            m_intermediates->falloffIntercept = std::pow(y1, 2) - m_intermediates->falloffSlope * m_parameters.tangentSpeedParabolicRoot;
            m_intermediates->falloffStartX = m_parameters.tangentSpeedParabolicRoot;
        }
    } else if (m_parameters.tangentSpeedParabolicRoot) {
        tangentInitialY = evaluateQuartic(m_parameters.tangentSpeedParabolicRoot);
        m_intermediates->falloffSlope = quarticSlope(m_parameters.tangentSpeedParabolicRoot);
        m_intermediates->falloffIntercept = std::pow(tangentInitialY, 2) - m_intermediates->falloffSlope *  m_parameters.tangentSpeedParabolicRoot;
        m_intermediates->tangentStartX = m_parameters.tangentSpeedParabolicRoot;
    }
}

float ScrollingAccelerationCurve::evaluateQuartic(float x) const
{
    return std::pow(m_parameters.gainQuartic * x, 4) + std::pow(m_parameters.gainCubic * x, 3) + std::pow(m_parameters.gainParabolic * x, 2) + m_parameters.gainLinear * x;
}

float ScrollingAccelerationCurve::accelerationFactor(float value)
{
    constexpr float frameRate = 67.f; // Why is this 67 and not 60?
    constexpr float cursorScale = 96.f / frameRate;

    computeIntermediateValuesIfNeeded();

    float multiplier;
    float deviceScale = m_parameters.resolution / frameRate;
    value /= deviceScale;

    if (value <= m_intermediates->tangentStartX)
        multiplier = evaluateQuartic(value);
    else if (value <= m_intermediates->falloffStartX && m_intermediates->tangentStartX == m_parameters.tangentSpeedLinear)
        multiplier = m_intermediates->tangentSlope * value + m_intermediates->tangentIntercept;
    else
        multiplier = std::sqrt(m_intermediates->falloffSlope * value + m_intermediates->falloffIntercept);

    return multiplier * cursorScale;
}

TextStream& operator<<(TextStream& ts, const ScrollingAccelerationCurve& curve)
{
    TextStream::GroupScope group(ts);

    ts << "ScrollingAccelerationCurve";

    ts.dumpProperty("gainLinear", curve.m_parameters.gainLinear);
    ts.dumpProperty("gainParabolic", curve.m_parameters.gainParabolic);
    ts.dumpProperty("gainCubic", curve.m_parameters.gainCubic);
    ts.dumpProperty("gainQuartic", curve.m_parameters.gainQuartic);
    ts.dumpProperty("tangentSpeedLinear", curve.m_parameters.tangentSpeedLinear);
    ts.dumpProperty("tangentSpeedParabolicRoot", curve.m_parameters.tangentSpeedParabolicRoot);
    ts.dumpProperty("resolution", curve.m_parameters.resolution);
    ts.dumpProperty("frameRate", curve.m_parameters.frameRate);

    return ts;
}

#if !PLATFORM(MAC)

std::optional<ScrollingAccelerationCurve> ScrollingAccelerationCurve::fromNativeWheelEvent(const NativeWebWheelEvent&)
{
    return std::nullopt;
}

#endif

} // namespace WebKit

#endif // ENABLE(MOMENTUM_EVENT_DISPATCHER)
