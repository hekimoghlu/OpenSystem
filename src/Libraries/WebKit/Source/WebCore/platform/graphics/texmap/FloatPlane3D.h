/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#include "FloatPoint3D.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPlane3D {
    WTF_MAKE_TZONE_ALLOCATED(FloatPlane3D);
public:
    FloatPlane3D(const FloatPoint3D&, const FloatPoint3D&);

    const FloatPoint3D& normal() const { return m_normal; }

    // Getter for the distance from the origin (plane constant d)
    float distanceConstant() const { return m_distanceConstant; }

    // Signed distance. The sign of the return value is positive
    // if the point is on the front side of the plane, negative if the
    // point is on the back side, and zero if the point is on the plane.
    float distanceToPoint(const FloatPoint3D& point) const
    {
        return m_normal.dot(point) - m_distanceConstant;
    }

private:
    FloatPoint3D m_normal;
    float m_distanceConstant;
};

} // namespace WebCore
