/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

namespace WTF {
class TextStream;
}

namespace WebCore {

class HitTestLocation;

class ClipRect {
public:
    ClipRect() = default;
    
    ClipRect(const LayoutRect& rect)
        : m_rect(rect)
    {
    }
    
    const LayoutRect& rect() const { return m_rect; }

    void reset() { m_rect = LayoutRect::infiniteRect(); }

    bool affectedByRadius() const { return m_affectedByRadius; }
    void setAffectedByRadius(bool affectedByRadius) { m_affectedByRadius = affectedByRadius; }
    
    friend bool operator==(const ClipRect&, const ClipRect&) = default;
    
    void intersect(const LayoutRect& other);
    void intersect(const ClipRect& other);
    void move(LayoutUnit x, LayoutUnit y) { m_rect.move(x, y); }
    void move(const LayoutSize& size) { m_rect.move(size); }
    void moveBy(const LayoutPoint& point) { m_rect.moveBy(point); }
    
    bool intersects(const LayoutRect&) const;
    bool intersects(const HitTestLocation&) const;

    bool isEmpty() const { return m_rect.isEmpty(); }
    bool isInfinite() const { return m_rect.isInfinite(); }

    void inflateX(LayoutUnit dx) { m_rect.inflateX(dx); }
    void inflateY(LayoutUnit dy) { m_rect.inflateY(dy); }
    void inflate(LayoutUnit d) { inflateX(d); inflateY(d); }
    
private:
    LayoutRect m_rect;
    bool m_affectedByRadius = false;
};

inline void ClipRect::intersect(const LayoutRect& other)
{
    if (other.isInfinite())
        return;
    if (isInfinite())
        m_rect = other;
    else
        m_rect.intersect(other);
}

inline void ClipRect::intersect(const ClipRect& other)
{
    intersect(other.rect());
    if (other.affectedByRadius())
        m_affectedByRadius = true;
}
    
inline bool ClipRect::intersects(const LayoutRect& rect) const
{
    if (isInfinite() || rect.isInfinite())
        return true;
    return m_rect.intersects(rect);
}

inline ClipRect intersection(const ClipRect& a, const ClipRect& b)
{
    ClipRect c = a;
    c.intersect(b);
    return c;
}

WTF::TextStream& operator<<(WTF::TextStream&, const ClipRect&);

} // namespace WebCore
