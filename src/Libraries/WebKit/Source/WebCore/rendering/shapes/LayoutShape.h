/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#include "LayoutRect.h"
#include "Path.h"
#include "StyleBasicShape.h"
#include "WritingMode.h"
#include <wtf/RefCounted.h>

namespace WebCore {

struct LineSegment {
    LineSegment()
        : logicalLeft(0)
        , logicalRight(0)
        , isValid(false)
    {
    }

    LineSegment(float logicalLeft, float logicalRight)
        : logicalLeft(logicalLeft)
        , logicalRight(logicalRight)
        , isValid(true)
    {
    }

    float logicalLeft;
    float logicalRight;
    bool isValid;
};

class Image;
class RoundedRect;

// A representation of a BasicShape that enables layout code to determine how to break a line up into segments
// that will fit within or around a shape. The line is defined by a pair of logical Y coordinates and the
// computed segments are returned as pairs of logical X coordinates. The BasicShape itself is defined in
// physical coordinates.

class LayoutShape : public RefCounted<LayoutShape> {
public:
    struct DisplayPaths {
        Path shape;
        Path marginShape;
    };

    static Ref<const LayoutShape> createShape(const Style::BasicShape&, const LayoutPoint& borderBoxOffset, const LayoutSize& logicalBoxSize, WritingMode, float margin);
    static Ref<const LayoutShape> createRasterShape(Image*, float threshold, const LayoutRect& imageRect, const LayoutRect& marginRect, WritingMode, float margin);
    static Ref<const LayoutShape> createBoxShape(const RoundedRect&, WritingMode, float margin);

    virtual ~LayoutShape() = default;

    virtual LayoutRect shapeMarginLogicalBoundingBox() const = 0;
    virtual bool isEmpty() const = 0;
    virtual LineSegment getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const = 0;

    bool lineOverlapsShapeMarginBounds(LayoutUnit lineTop, LayoutUnit lineHeight) const { return lineOverlapsBoundingBox(lineTop, lineHeight, shapeMarginLogicalBoundingBox()); }

    virtual void buildDisplayPaths(DisplayPaths&) const = 0;

protected:
    float shapeMargin() const { return m_margin; }
    WritingMode writingMode() const { return m_writingMode; }

private:
    bool lineOverlapsBoundingBox(LayoutUnit lineTop, LayoutUnit lineHeight, const LayoutRect& rect) const
    {
        if (rect.isEmpty())
            return false;
        return (lineTop < rect.maxY() && lineTop + lineHeight > rect.y()) || (!lineHeight && lineTop == rect.y());
    }

    WritingMode m_writingMode;
    float m_margin;
};

} // namespace WebCore
