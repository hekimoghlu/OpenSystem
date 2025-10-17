/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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

namespace WebCore {
namespace Layout {

class FlexRect {
public:
    FlexRect() = default;
    struct Margins {
        LayoutUnit start;
        LayoutUnit end;
    };
    FlexRect(LayoutUnit top, LayoutUnit left, LayoutUnit width, LayoutUnit height, Margins mainAxisMargins, Margins crossAxisMargins);
    FlexRect(const LayoutRect&, Margins mainAxisMargins, Margins crossAxisMargins);

    LayoutUnit top() const;
    LayoutUnit left() const;
    LayoutPoint topLeft() const;

    LayoutUnit bottom() const;
    LayoutUnit right() const;        

    LayoutUnit width() const;
    LayoutUnit height() const;
    LayoutSize size() const;

    LayoutUnit marginTop() const { return m_crossAxisMargins.start; }
    LayoutUnit marginBottom() const { return m_crossAxisMargins.end; }
    LayoutUnit marginLeft() const { return m_mainAxisMargins.start; }
    LayoutUnit marginRight() const { return m_mainAxisMargins.end; }

    void setTop(LayoutUnit);
    void setBottom(LayoutUnit);
    void setLeft(LayoutUnit);
    void setRight(LayoutUnit);
    void setTopLeft(const LayoutPoint&);
    void setWidth(LayoutUnit);
    void setHeight(LayoutUnit);

    void setMarginTop(LayoutUnit marginTop) { m_crossAxisMargins.start = marginTop; }
    void setMarginBottom(LayoutUnit marginBottom) { m_crossAxisMargins.end = marginBottom; }
    void setMarginLeft(LayoutUnit marginLeft) { m_mainAxisMargins.start = marginLeft; }
    void setMarginRight(LayoutUnit marginRight) { m_mainAxisMargins.end = marginRight; }

    void moveHorizontally(LayoutUnit);
    void moveVertically(LayoutUnit);

    bool isEmpty() const;

    operator LayoutRect() const;

private:
#if ASSERT_ENABLED
    void invalidateTop() { m_hasValidTop = false; }
    void invalidateLeft() { m_hasValidLeft = false; }
    void invalidateWidth() { m_hasValidWidth = false; }
    void invalidateHeight() { m_hasValidHeight = false; }
    void invalidatePosition();

    bool hasValidPosition() const { return m_hasValidTop && m_hasValidLeft; }
    bool hasValidSize() const { return m_hasValidWidth && m_hasValidHeight; }
    bool hasValidGeometry() const { return hasValidPosition() && hasValidSize(); }

    void setHasValidPosition();
    void setHasValidSize();

    bool m_hasValidTop { false };
    bool m_hasValidLeft { false };
    bool m_hasValidWidth { false };
    bool m_hasValidHeight { false };
#endif // ASSERT_ENABLED
    LayoutRect m_rect;
    Margins m_mainAxisMargins;
    Margins m_crossAxisMargins;
};

inline FlexRect::FlexRect(LayoutUnit top, LayoutUnit left, LayoutUnit width, LayoutUnit height, Margins mainAxisMargins, Margins crossAxisMargins)
    : m_rect(left, top, width, height)
    , m_mainAxisMargins(mainAxisMargins)
    , m_crossAxisMargins(crossAxisMargins)
{
#if ASSERT_ENABLED
    m_hasValidTop = true;
    m_hasValidLeft = true;
    m_hasValidWidth = true;
    m_hasValidHeight = true;
#endif
}

inline FlexRect::FlexRect(const LayoutRect& rect, Margins mainAxisMargins, Margins crossAxisMargins)
    : FlexRect(rect.y(), rect.x(), rect.width(), rect.height(), mainAxisMargins, crossAxisMargins)
{
}

#if ASSERT_ENABLED
inline void FlexRect::invalidatePosition()
{
    invalidateTop();
    invalidateLeft();
}

inline void FlexRect::setHasValidPosition()
{
    m_hasValidTop = true;
    m_hasValidLeft = true;
}

inline void FlexRect::setHasValidSize()
{
    m_hasValidWidth = true;
    m_hasValidHeight = true;
}
#endif // ASSERT_ENABLED

inline LayoutUnit FlexRect::top() const
{
    ASSERT(m_hasValidTop);
    return m_rect.y();
}

inline LayoutUnit FlexRect::left() const
{
    ASSERT(m_hasValidLeft);
    return m_rect.x();
}

inline LayoutUnit FlexRect::bottom() const
{
    ASSERT(m_hasValidTop && m_hasValidHeight);
    return m_rect.maxY();
}

inline LayoutUnit FlexRect::right() const
{
    ASSERT(m_hasValidLeft && m_hasValidWidth);
    return m_rect.maxX();
}

inline LayoutPoint FlexRect::topLeft() const
{
    ASSERT(hasValidPosition());
    return m_rect.minXMinYCorner();
}

inline LayoutSize FlexRect::size() const
{
    ASSERT(hasValidSize());
    return m_rect.size();
}

inline LayoutUnit FlexRect::width() const
{
    ASSERT(m_hasValidWidth);
    return m_rect.width();
}

inline LayoutUnit FlexRect::height() const
{
    ASSERT(m_hasValidHeight);
    return m_rect.height();
}

inline void FlexRect::setTopLeft(const LayoutPoint& topLeft)
{
#if ASSERT_ENABLED
    setHasValidPosition();
#endif
    m_rect.setLocation(topLeft);
}

inline void FlexRect::setTop(LayoutUnit top)
{
#if ASSERT_ENABLED
    m_hasValidTop = true;
#endif
    m_rect.setY(top);
}

inline void FlexRect::setBottom(LayoutUnit bottom)
{
#if ASSERT_ENABLED
    m_hasValidTop = true;
    m_hasValidHeight = true;
#endif
    m_rect.shiftMaxYEdgeTo(bottom);
}

inline void FlexRect::setLeft(LayoutUnit left)
{
#if ASSERT_ENABLED
    m_hasValidLeft = true;
#endif
    m_rect.setX(left);
}

inline void FlexRect::setRight(LayoutUnit right)
{
#if ASSERT_ENABLED
    m_hasValidLeft = true;
    m_hasValidWidth = true;
#endif
    m_rect.shiftMaxXEdgeTo(right);
}

inline void FlexRect::setWidth(LayoutUnit width)
{
#if ASSERT_ENABLED
    m_hasValidWidth = true;
#endif
    m_rect.setWidth(width);
}

inline void FlexRect::setHeight(LayoutUnit height)
{
#if ASSERT_ENABLED
    m_hasValidHeight = true;
#endif
    m_rect.setHeight(height);
}

inline void FlexRect::moveHorizontally(LayoutUnit offset)
{
    ASSERT(m_hasValidLeft);
    m_rect.move(LayoutSize { offset, 0 });
}

inline void FlexRect::moveVertically(LayoutUnit offset)
{
    ASSERT(m_hasValidTop);
    m_rect.move(LayoutSize { 0, offset });
}

inline bool FlexRect::isEmpty() const
{
    ASSERT(hasValidGeometry());
    return m_rect.isEmpty();
}

inline FlexRect::operator LayoutRect() const
{
    ASSERT(hasValidGeometry()); 
    return m_rect;
}

}
}
