/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore
{
// RenderOverflow is a class for tracking content that spills out of a box.  This class is used by RenderBox and
// LegacyInlineFlowBox.
//
// There are two types of overflow: layout overflow (which is expected to be reachable via scrolling mechanisms) and
// visual overflow (which is not expected to be reachable via scrolling mechanisms).
//
// Layout overflow examples include other boxes that spill out of our box,  For example, in the inline case a tall image
// could spill out of a line box. 
    
// Examples of visual overflow are shadows, text stroke, outline (and eventually border-image).

// This object is allocated only when some of these fields have non-default values in the owning box.
class RenderOverflow : public RefCounted<RenderOverflow> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RenderOverflow);
    WTF_MAKE_NONCOPYABLE(RenderOverflow);
public:
    RenderOverflow(const LayoutRect& layoutRect, const LayoutRect& visualRect) 
        : m_layoutOverflow(layoutRect)
        , m_visualOverflow(visualRect)
    {
    }
   
    const LayoutRect layoutOverflowRect() const { return m_layoutOverflow; }
    const LayoutRect visualOverflowRect() const { return m_visualOverflow; }
    
    void move(LayoutUnit dx, LayoutUnit dy);
    
    void addLayoutOverflow(const LayoutRect&);
    void addVisualOverflow(const LayoutRect&);

    void setLayoutOverflow(const LayoutRect&);
    void setVisualOverflow(const LayoutRect&);

    LayoutUnit layoutClientAfterEdge() const { return m_layoutClientAfterEdge; }
    void setLayoutClientAfterEdge(LayoutUnit clientAfterEdge) { m_layoutClientAfterEdge = clientAfterEdge; }

private:
    LayoutRect m_layoutOverflow;
    LayoutRect m_visualOverflow;

    LayoutUnit m_layoutClientAfterEdge;
};

inline void RenderOverflow::move(LayoutUnit dx, LayoutUnit dy)
{
    m_layoutOverflow.move(dx, dy);
    m_visualOverflow.move(dx, dy);
}

inline void RenderOverflow::addLayoutOverflow(const LayoutRect& rect)
{
    LayoutUnit maxX = std::max(rect.maxX(), m_layoutOverflow.maxX());
    LayoutUnit maxY = std::max(rect.maxY(), m_layoutOverflow.maxY());
    LayoutUnit minX = std::min(rect.x(), m_layoutOverflow.x());
    LayoutUnit minY = std::min(rect.y(), m_layoutOverflow.y());
    // In case the width/height is larger than LayoutUnit can represent, fix the right/bottom edge and shift the top/left ones
    m_layoutOverflow.setWidth(maxX - minX);
    m_layoutOverflow.setHeight(maxY - minY);
    m_layoutOverflow.setX(maxX - m_layoutOverflow.width());
    m_layoutOverflow.setY(maxY - m_layoutOverflow.height());
}

inline void RenderOverflow::addVisualOverflow(const LayoutRect& rect)
{
    LayoutUnit maxX = std::max(rect.maxX(), m_visualOverflow.maxX());
    LayoutUnit maxY = std::max(rect.maxY(), m_visualOverflow.maxY());
    m_visualOverflow.setX(std::min(rect.x(), m_visualOverflow.x()));
    m_visualOverflow.setY(std::min(rect.y(), m_visualOverflow.y()));
    m_visualOverflow.setWidth(maxX - m_visualOverflow.x());
    m_visualOverflow.setHeight(maxY - m_visualOverflow.y());
}

inline void RenderOverflow::setLayoutOverflow(const LayoutRect& rect)
{
    m_layoutOverflow = rect;
}

inline void RenderOverflow::setVisualOverflow(const LayoutRect& rect)
{
    m_visualOverflow = rect;
}

} // namespace WebCore
