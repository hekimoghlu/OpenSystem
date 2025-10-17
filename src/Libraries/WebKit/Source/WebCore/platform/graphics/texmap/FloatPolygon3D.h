/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
#include "TransformationMatrix.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class FloatPlane3D;
class FloatRect;

// FloatPolygon3D represents planar polygon in 3D space

class FloatPolygon3D {
    WTF_MAKE_TZONE_ALLOCATED(FloatPolygon3D);
public:
    FloatPolygon3D() = default;
    FloatPolygon3D(const FloatRect&, const TransformationMatrix&);

    const FloatPoint3D& vertexAt(unsigned index) const { return m_vertices[index]; }
    unsigned numberOfVertices() const { return m_vertices.size(); }

    const FloatPoint3D& normal() const { return m_normal; }

    std::pair<FloatPolygon3D, FloatPolygon3D> split(const FloatPlane3D&) const;

private:
    FloatPolygon3D(Vector<FloatPoint3D>&&, const FloatPoint3D&);

    Vector<FloatPoint3D> m_vertices;
    FloatPoint3D m_normal { 0, 0, 1 };
};

} // namespace WebCore
