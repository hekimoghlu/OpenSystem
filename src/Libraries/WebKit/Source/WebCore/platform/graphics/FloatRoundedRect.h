/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#ifndef FloatRoundedRect_h
#define FloatRoundedRect_h

#include "FloatRect.h"
#include "FloatSize.h"
#include "Region.h"
#include "RoundedRect.h"
#include <wtf/TZoneMalloc.h>

#if USE(SKIA)
class SkRRect;
#endif

namespace WebCore {

class FloatRoundedRect {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FloatRoundedRect, WEBCORE_EXPORT);
public:
    class Radii {
        WTF_MAKE_TZONE_ALLOCATED(Radii);
    public:
        Radii() = default;
        Radii(const FloatSize& topLeft, const FloatSize& topRight, const FloatSize& bottomLeft, const FloatSize& bottomRight)
            : m_topLeft(topLeft)
            , m_topRight(topRight)
            , m_bottomLeft(bottomLeft)
            , m_bottomRight(bottomRight)
        {
        }

        Radii(const RoundedRect::Radii& intRadii)
            : m_topLeft(intRadii.topLeft())
            , m_topRight(intRadii.topRight())
            , m_bottomLeft(intRadii.bottomLeft())
            , m_bottomRight(intRadii.bottomRight())
        {
        }

        explicit Radii(float uniformRadius)
            : m_topLeft(uniformRadius, uniformRadius)
            , m_topRight(uniformRadius, uniformRadius)
            , m_bottomLeft(uniformRadius, uniformRadius)
            , m_bottomRight(uniformRadius, uniformRadius)
        {
        }

        explicit Radii(float uniformRadiusWidth, float uniformRadiusHeight)
            : m_topLeft(uniformRadiusWidth, uniformRadiusHeight)
            , m_topRight(uniformRadiusWidth, uniformRadiusHeight)
            , m_bottomLeft(uniformRadiusWidth, uniformRadiusHeight)
            , m_bottomRight(uniformRadiusWidth, uniformRadiusHeight)
        {
        }

        void setTopLeft(const FloatSize& size) { m_topLeft = size; }
        void setTopRight(const FloatSize& size) { m_topRight = size; }
        void setBottomLeft(const FloatSize& size) { m_bottomLeft = size; }
        void setBottomRight(const FloatSize& size) { m_bottomRight = size; }
        const FloatSize& topLeft() const { return m_topLeft; }
        const FloatSize& topRight() const { return m_topRight; }
        const FloatSize& bottomLeft() const { return m_bottomLeft; }
        const FloatSize& bottomRight() const { return m_bottomRight; }

        bool isZero() const;
        bool hasEvenCorners() const;
        bool isUniformCornerRadius() const; // Including no radius.

        void scale(float factor);
        void scale(float horizontalFactor, float verticalFactor);
        void expandEvenIfZero(float size);
        void expand(float topWidth, float bottomWidth, float leftWidth, float rightWidth);
        void expand(float size) { expand(size, size, size, size); }
        void shrink(float topWidth, float bottomWidth, float leftWidth, float rightWidth) { expand(-topWidth, -bottomWidth, -leftWidth, -rightWidth); }
        void shrink(float size) { shrink(size, size, size, size); }

        friend bool operator==(const Radii&, const Radii&) = default;

    private:
        FloatSize m_topLeft;
        FloatSize m_topRight;
        FloatSize m_bottomLeft;
        FloatSize m_bottomRight;
    };

    WEBCORE_EXPORT explicit FloatRoundedRect(const FloatRect& = FloatRect(), const Radii& = Radii());
    WEBCORE_EXPORT FloatRoundedRect(const FloatRect&, const FloatSize& topLeft, const FloatSize& topRight, const FloatSize& bottomLeft, const FloatSize& bottomRight);
    explicit FloatRoundedRect(const RoundedRect&);
    FloatRoundedRect(float x, float y, float width, float height);

    const FloatRect& rect() const { return m_rect; }
    const Radii& radii() const { return m_radii; }
    bool isRounded() const { return !m_radii.isZero(); }
    bool isEmpty() const { return m_rect.isEmpty(); }

    void setRect(const FloatRect& rect) { m_rect = rect; }
    void setLocation(FloatPoint location) { m_rect.setLocation(location); }
    void setRadii(const Radii& radii) { m_radii = radii; }

    void move(const FloatSize& size) { m_rect.move(size); }
    void inflate(float size) { m_rect.inflate(size);  }
    void expandRadii(float size) { m_radii.expand(size); }
    void shrinkRadii(float size) { m_radii.shrink(size); }
    void inflateWithRadii(float size);
    void adjustRadii();

    FloatRect topLeftCorner() const
    {
        return FloatRect(m_rect.x(), m_rect.y(), m_radii.topLeft().width(), m_radii.topLeft().height());
    }
    FloatRect topRightCorner() const
    {
        return FloatRect(m_rect.maxX() - m_radii.topRight().width(), m_rect.y(), m_radii.topRight().width(), m_radii.topRight().height());
    }
    FloatRect bottomLeftCorner() const
    {
        return FloatRect(m_rect.x(), m_rect.maxY() - m_radii.bottomLeft().height(), m_radii.bottomLeft().width(), m_radii.bottomLeft().height());
    }
    FloatRect bottomRightCorner() const
    {
        return FloatRect(m_rect.maxX() - m_radii.bottomRight().width(), m_rect.maxY() - m_radii.bottomRight().height(), m_radii.bottomRight().width(), m_radii.bottomRight().height());
    }

    bool isRenderable() const;
    bool xInterceptsAtY(float y, float& minXIntercept, float& maxXIntercept) const;

    bool intersectionIsRectangular(const FloatRect&) const;

    friend bool operator==(const FloatRoundedRect&, const FloatRoundedRect&) = default;

#if USE(SKIA)
    FloatRoundedRect(const SkRRect&);
    operator SkRRect() const;
#endif

private:
    FloatRect m_rect;
    Radii m_radii;
};

inline float calcBorderRadiiConstraintScaleFor(const FloatRect& rect, const FloatRoundedRect::Radii& radii)
{
    // Constrain corner radii using CSS3 rules:
    // http://www.w3.org/TR/css3-background/#the-border-radius

    float factor = 1;
    float radiiSum;

    // top
    radiiSum = radii.topLeft().width() + radii.topRight().width(); // Casts to avoid integer overflow.
    if (radiiSum > rect.width())
        factor = std::min(rect.width() / radiiSum, factor);

    // bottom
    radiiSum = radii.bottomLeft().width() + radii.bottomRight().width();
    if (radiiSum > rect.width())
        factor = std::min(rect.width() / radiiSum, factor);

    // left
    radiiSum = radii.topLeft().height() + radii.bottomLeft().height();
    if (radiiSum > rect.height())
        factor = std::min(rect.height() / radiiSum, factor);

    // right
    radiiSum = radii.topRight().height() + radii.bottomRight().height();
    if (radiiSum > rect.height())
        factor = std::min(rect.height() / radiiSum, factor);

    ASSERT(factor <= 1);
    return factor;
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FloatRoundedRect&);

// Snip away rectangles from corners, roughly one per step length of arc.
WEBCORE_EXPORT Region approximateAsRegion(const FloatRoundedRect&, unsigned stepLength = 20);

} // namespace WebCore

#endif // FloatRoundedRect_h
