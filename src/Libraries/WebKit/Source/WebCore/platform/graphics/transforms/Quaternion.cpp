/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include "Quaternion.h"

#include <cmath>

namespace WebCore {

// Perform a spherical linear interpolation between the two
// passed quaternions with 0 <= t <= 1.

Quaternion Quaternion::slerp(const Quaternion& other, double t)
{
    const double kEpsilon = 1e-5;
    Quaternion copy = *this;

    double cosHalfAngle = copy.x * other.x + copy.y * other.y + copy.z * other.z + copy.w * other.w;

    if (cosHalfAngle < 0.0) {
        copy.x = -copy.x;
        copy.y = -copy.y;
        copy.z = -copy.z;
        copy.w = -copy.w;
        cosHalfAngle = -cosHalfAngle;
    }

    if (cosHalfAngle > 1)
        cosHalfAngle = 1;

    double sinHalfAngle = std::sqrt(1.0 - cosHalfAngle * cosHalfAngle);
    if (sinHalfAngle < kEpsilon) {
        // Quaternions share common axis and angle.
        return *this;
    }

    double halfAngle = std::acos(cosHalfAngle);
    double scale = std::sin((1 - t) * halfAngle) / sinHalfAngle;
    double invscale = std::sin(t * halfAngle) / sinHalfAngle;

    return { copy.x * scale + other.x * invscale, copy.y * scale + other.y * invscale, copy.z * scale + other.z * invscale, copy.w * scale + other.w * invscale };
}

// Compute quaternion multiplication
Quaternion Quaternion::accumulate(const Quaternion& other)
{
    return { w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w,
        w * other.w - x * other.x - y * other.y - z * other.z };
}

Quaternion Quaternion::interpolate(const Quaternion& other, double progress, CompositeOperation compositeOperation)
{
    if (compositeOperation == CompositeOperation::Accumulate)
        return accumulate(other);
    return slerp(other, progress);
}

} // namespace WebCore
