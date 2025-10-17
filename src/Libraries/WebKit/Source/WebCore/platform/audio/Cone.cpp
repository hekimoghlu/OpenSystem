/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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

#if ENABLE(WEB_AUDIO)

#include "Cone.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ConeEffect);

ConeEffect::ConeEffect() = default;

double ConeEffect::gain(FloatPoint3D sourcePosition, FloatPoint3D sourceOrientation, FloatPoint3D listenerPosition) const
{
    if (sourceOrientation.isZero() || ((m_innerAngle == 360.0) && (m_outerAngle == 360.0)))
        return 1.0; // no cone specified - unity gain

    // Normalized source-listener vector
    FloatPoint3D sourceToListener = listenerPosition - sourcePosition;
    sourceToListener.normalize();

    sourceOrientation.normalize();

    // Due to precision issues, the dot product may be very slightly outside the range [-1.0, 1.0], which would
    // acos() to return NaN. For this reason, we have to make sure we clamp the dot product before calling acos().
    auto dotProduct = clampTo(sourceToListener.dot(sourceOrientation), -1.0, 1.0);

    // Angle between the source orientation vector and the source-listener vector
    double angle = rad2deg(acos(dotProduct));
    double absAngle = std::abs(angle);

    // Divide by 2.0 here since API is entire angle (not half-angle)
    double absInnerAngle = std::abs(m_innerAngle) / 2.0;
    double absOuterAngle = std::abs(m_outerAngle) / 2.0;
    double gain = 1.0;

    if (absAngle <= absInnerAngle) {
        // No attenuation
        gain = 1.0;
    } else if (absAngle >= absOuterAngle) {
        // Max attenuation
        gain = m_outerGain;
    } else {
        // Between inner and outer cones
        // inner -> outer, x goes from 0 -> 1
        double x = (absAngle - absInnerAngle) / (absOuterAngle - absInnerAngle);
        gain = (1.0 - x) + m_outerGain * x;
    }

    return gain;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
