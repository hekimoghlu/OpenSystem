/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

#include "FloatRect.h"
#include "FloatSize.h"
#include "LayoutShape.h"
#include <wtf/Assertions.h>

namespace WebCore {

class RectangleLayoutShape final : public LayoutShape {
public:
    RectangleLayoutShape(const FloatRect& bounds, const FloatSize& radii, float boxLogicalWidth)
        : m_bounds(bounds)
        , m_radii(radii)
        , m_boxLogicalWidth(boxLogicalWidth)
    {
    }

    LayoutRect shapeMarginLogicalBoundingBox() const override { return static_cast<LayoutRect>(shapeMarginBounds()); }
    bool isEmpty() const override { return m_bounds.isEmpty(); }
    LineSegment getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const override;

    void buildDisplayPaths(DisplayPaths&) const override;

private:
    FloatRect shapeMarginBounds() const;

    float rx() const { return m_radii.width(); }
    float ry() const { return m_radii.height(); }
    float x() const { return m_bounds.x(); }
    float y() const { return m_bounds.y(); }
    float width() const { return m_bounds.width(); }
    float height() const { return m_bounds.height(); }

    FloatRect m_bounds;
    FloatSize m_radii;
    float m_boxLogicalWidth { 0.f };
};

} // namespace WebCore
