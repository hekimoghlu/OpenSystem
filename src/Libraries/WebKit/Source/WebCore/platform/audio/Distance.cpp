/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "Distance.h"
#include <wtf/TZoneMallocInlines.h>

#include <algorithm>
#include <math.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DistanceEffect);

DistanceEffect::DistanceEffect() = default;

double DistanceEffect::gain(double distance) const
{
    // don't go beyond maximum distance
    distance = std::min(distance, m_maxDistance);

    // if clamped, don't get closer than reference distance
    if (m_isClamped)
        distance = std::max(distance, m_refDistance);

    switch (m_model) {
    case DistanceModelType::Linear:
        return linearGain(distance);
    case DistanceModelType::Inverse:
        return inverseGain(distance);
    case DistanceModelType::Exponential:
        return exponentialGain(distance);
    }
    ASSERT_NOT_REACHED();
    return 0.0;
}

double DistanceEffect::linearGain(double distance) const
{
    auto clampedRolloffFactor = std::clamp(m_rolloffFactor, 0.0, 1.0);
    // We want a gain that decreases linearly from m_refDistance to
    // m_maxDistance. The gain is 1 at m_refDistance.
    return (1.0 - clampedRolloffFactor * (distance - m_refDistance) / (m_maxDistance - m_refDistance));
}

double DistanceEffect::inverseGain(double distance) const
{
    return m_refDistance / (m_refDistance + m_rolloffFactor * (distance - m_refDistance));
}

double DistanceEffect::exponentialGain(double distance) const
{
    return pow(distance / m_refDistance, -m_rolloffFactor);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
