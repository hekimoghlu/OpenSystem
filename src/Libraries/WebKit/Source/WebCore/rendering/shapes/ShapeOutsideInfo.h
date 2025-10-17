/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

#include "LayoutShape.h"
#include "LayoutSize.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderBlockFlow;
class RenderBox;
class StyleImage;
class FloatingObject;

Ref<const LayoutShape> makeShapeForShapeOutside(const RenderBox&);

class ShapeOutsideDeltas final {
    WTF_MAKE_TZONE_ALLOCATED(ShapeOutsideDeltas);
public:
    ShapeOutsideDeltas()
        : m_lineOverlapsShape(false)
        , m_isValid(false)
    {
    }

    ShapeOutsideDeltas(LayoutUnit leftMarginBoxDelta, LayoutUnit rightMarginBoxDelta, bool lineOverlapsShape, LayoutUnit borderBoxLineTop, LayoutUnit lineHeight)
        : m_leftMarginBoxDelta(leftMarginBoxDelta)
        , m_rightMarginBoxDelta(rightMarginBoxDelta)
        , m_borderBoxLineTop(borderBoxLineTop)
        , m_lineHeight(lineHeight)
        , m_lineOverlapsShape(lineOverlapsShape)
        , m_isValid(true)
    {
    }

    bool isForLine(LayoutUnit borderBoxLineTop, LayoutUnit lineHeight)
    {
        return m_isValid && m_borderBoxLineTop == borderBoxLineTop && m_lineHeight == lineHeight;
    }

    bool isValid() { return m_isValid; }
    LayoutUnit leftMarginBoxDelta() { ASSERT(m_isValid); return m_leftMarginBoxDelta; }
    LayoutUnit rightMarginBoxDelta() { ASSERT(m_isValid); return m_rightMarginBoxDelta; }
    bool lineOverlapsShape() { ASSERT(m_isValid); return m_lineOverlapsShape; }

private:
    LayoutUnit m_leftMarginBoxDelta;
    LayoutUnit m_rightMarginBoxDelta;
    LayoutUnit m_borderBoxLineTop;
    LayoutUnit m_lineHeight;
    unsigned m_lineOverlapsShape : 1;
    unsigned m_isValid : 1;
};

class ShapeOutsideInfo final {
    WTF_MAKE_TZONE_ALLOCATED(ShapeOutsideInfo);
public:
    ShapeOutsideInfo(const RenderBox& renderer)
        : m_renderer(renderer)
    {
    }

    static bool isEnabledFor(const RenderBox&);

    ShapeOutsideDeltas computeDeltasForContainingBlockLine(const RenderBlockFlow&, const FloatingObject&, LayoutUnit lineTop, LayoutUnit lineHeight);

    void invalidateForSizeChangeIfNeeded();

    LayoutUnit shapeLogicalBottom() const { return computedShape().shapeMarginLogicalBoundingBox().maxY(); }

    void markShapeAsDirty() { m_shape = nullptr; }
    bool isShapeDirty() { return !m_shape; }

    LayoutRect computedShapePhysicalBoundingBox() const;
    FloatPoint shapeToRendererPoint(const FloatPoint&) const;

    const LayoutShape& computedShape() const;

private:
    LayoutUnit logicalTopOffset() const;
    LayoutUnit logicalLeftOffset() const;

    const RenderBox& m_renderer;

    mutable RefPtr<const LayoutShape> m_shape;
    LayoutSize m_cachedShapeLogicalSize;

    ShapeOutsideDeltas m_shapeOutsideDeltas;
};

}
