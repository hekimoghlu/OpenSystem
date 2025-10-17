/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

#include "LayoutBox.h"
#include "LayoutBoxGeometry.h"
#include "LayoutPoint.h"
#include "LayoutUnits.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

namespace Layout {

class FloatAvoider {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FloatAvoider);
public:
    FloatAvoider(LayoutPoint absoluteTopLeft, LayoutUnit borderBoxWidth, const BoxGeometry::Edges& margin, BoxGeometry::HorizontalEdges containingBlockAbsoluteContentBox, bool isFloatingPositioned, bool isLeftAligned);
    virtual ~FloatAvoider() = default;

    void setInlineStart(LayoutUnit);
    void setBlockStart(LayoutUnit);
    void resetInlineStart() { m_absoluteTopLeft.setX(initialInlineStart()); }

    bool overflowsContainingBlock() const;

    LayoutUnit blockStart() const;
    LayoutUnit inlineStart() const;
    LayoutUnit inlineEnd() const;

    bool isStartAligned() const { return m_isStartAligned; }

private:
    LayoutUnit borderBoxWidth() const { return m_borderBoxWidth; }
    LayoutUnit initialInlineStart() const;

    LayoutUnit marginBefore() const { return m_margin.vertical.before; }
    LayoutUnit marginAfter() const { return m_margin.vertical.after; }
    LayoutUnit marginStart() const { return m_margin.horizontal.start; }
    LayoutUnit marginEnd() const { return m_margin.horizontal.end; }
    LayoutUnit marginBoxWidth() const { return marginStart() + borderBoxWidth() + marginEnd(); }

    bool isFloatingBox() const { return m_isFloatingPositioned; }

private:
    // These coordinate values are relative to the formatting root's border box.
    LayoutPoint m_absoluteTopLeft;
    // Note that float avoider should work with no height value.
    LayoutUnit m_borderBoxWidth;
    BoxGeometry::Edges m_margin;
    BoxGeometry::HorizontalEdges m_containingBlockAbsoluteContentBox;
    bool m_isFloatingPositioned { true };
    bool m_isStartAligned { true };
};

inline LayoutUnit FloatAvoider::blockStart() const
{
    auto blockStart = m_absoluteTopLeft.y();
    if (isFloatingBox())
        blockStart -= marginBefore();
    return blockStart;
}

inline LayoutUnit FloatAvoider::inlineStart() const
{
    auto inlineStart = m_absoluteTopLeft.x();
    if (isFloatingBox())
        inlineStart -= marginStart();
    return inlineStart;
}

inline LayoutUnit FloatAvoider::inlineEnd() const
{
    auto inlineEnd = inlineStart() + borderBoxWidth();
    if (isFloatingBox())
        inlineEnd += marginEnd();
    return inlineEnd;
}

}
}
