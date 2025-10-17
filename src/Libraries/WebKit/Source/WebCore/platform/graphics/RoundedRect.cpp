/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "config.h"
#include "RoundedRect.h"

#include "FloatQuad.h"
#include "FloatRoundedRect.h"
#include "GeometryUtilities.h"
#include "LayoutRect.h"
#include "LayoutUnit.h"
#include "Region.h"
#include <algorithm>
#include <wtf/MathExtras.h>

namespace WebCore {

bool RoundedRect::Radii::isZero() const
{
    return m_topLeft.isZero() && m_topRight.isZero() && m_bottomLeft.isZero() && m_bottomRight.isZero();
}

void RoundedRect::Radii::scale(float factor)
{
    if (factor == 1)
        return;

    // If either radius on a corner becomes zero, reset both radii on that corner.
    m_topLeft.scale(factor);
    if (!m_topLeft.width() || !m_topLeft.height())
        m_topLeft = LayoutSize();
    m_topRight.scale(factor);
    if (!m_topRight.width() || !m_topRight.height())
        m_topRight = LayoutSize();
    m_bottomLeft.scale(factor);
    if (!m_bottomLeft.width() || !m_bottomLeft.height())
        m_bottomLeft = LayoutSize();
    m_bottomRight.scale(factor);
    if (!m_bottomRight.width() || !m_bottomRight.height())
        m_bottomRight = LayoutSize();
}

void RoundedRect::Radii::expand(LayoutUnit topWidth, LayoutUnit bottomWidth, LayoutUnit leftWidth, LayoutUnit rightWidth)
{
    if (m_topLeft.width() > 0 && m_topLeft.height() > 0) {
        m_topLeft.setWidth(std::max<LayoutUnit>(0, m_topLeft.width() + leftWidth));
        m_topLeft.setHeight(std::max<LayoutUnit>(0, m_topLeft.height() + topWidth));
    }
    if (m_topRight.width() > 0 && m_topRight.height() > 0) {
        m_topRight.setWidth(std::max<LayoutUnit>(0, m_topRight.width() + rightWidth));
        m_topRight.setHeight(std::max<LayoutUnit>(0, m_topRight.height() + topWidth));
    }
    if (m_bottomLeft.width() > 0 && m_bottomLeft.height() > 0) {
        m_bottomLeft.setWidth(std::max<LayoutUnit>(0, m_bottomLeft.width() + leftWidth));
        m_bottomLeft.setHeight(std::max<LayoutUnit>(0, m_bottomLeft.height() + bottomWidth));
    }
    if (m_bottomRight.width() > 0 && m_bottomRight.height() > 0) {
        m_bottomRight.setWidth(std::max<LayoutUnit>(0, m_bottomRight.width() + rightWidth));
        m_bottomRight.setHeight(std::max<LayoutUnit>(0, m_bottomRight.height() + bottomWidth));
    }
}

void RoundedRect::inflateWithRadii(LayoutUnit amount)
{
    LayoutRect old = m_rect;

    if (amount < 0) {
        m_rect.inflateX(std::max(-m_rect.width() / 2, amount));
        m_rect.inflateY(std::max(-m_rect.height() / 2, amount));
    } else
        m_rect.inflate(amount);

    // Considering the inflation factor of shorter size to scale the radii seems appropriate here
    float factor;
    if (m_rect.width() < m_rect.height())
        factor = old.width() ? (float)m_rect.width() / old.width() : 0.0f;
    else
        factor = old.height() ? (float)m_rect.height() / old.height() : 0.0f;

    m_radii.scale(factor);
}

void RoundedRect::Radii::setRadiiForEdges(const RoundedRect::Radii& radii, RectEdges<bool> includeEdges)
{
    if (includeEdges.top()) {
        if (includeEdges.left())
            m_topLeft = radii.topLeft();
        if (includeEdges.right())
            m_topRight = radii.topRight();
    }
    if (includeEdges.bottom()) {
        if (includeEdges.left())
            m_bottomLeft = radii.bottomLeft();
        if (includeEdges.right())
            m_bottomRight = radii.bottomRight();
    }
}

bool RoundedRect::Radii::areRenderableInRect(const LayoutRect& rect) const
{
    return topLeft().width() >= 0 && topLeft().height() >= 0
        && bottomLeft().width() >= 0 && bottomLeft().height() >= 0
        && topRight().width() >= 0 && topRight().height() >= 0
        && bottomRight().width() >= 0 && bottomRight().height() >= 0
        && topLeft().width() + topRight().width() <= rect.width()
        && bottomLeft().width() + bottomRight().width() <= rect.width()
        && topLeft().height() + bottomLeft().height() <= rect.height()
        && topRight().height() + bottomRight().height() <= rect.height();
}

void RoundedRect::Radii::makeRenderableInRect(const LayoutRect& rect)
{
    auto maxRadiusWidth = std::max(topLeft().width() + topRight().width(), bottomLeft().width() + bottomRight().width());
    auto maxRadiusHeight = std::max(topLeft().height() + bottomLeft().height(), topRight().height() + bottomRight().height());

    if (maxRadiusWidth <= 0 || maxRadiusHeight <= 0) {
        scale(0);
        return;
    }

    float widthRatio = static_cast<float>(rect.width()) / maxRadiusWidth;
    float heightRatio = static_cast<float>(rect.height()) / maxRadiusHeight;
    scale(widthRatio < heightRatio ? widthRatio : heightRatio);

}

RoundedRect::RoundedRect(LayoutUnit x, LayoutUnit y, LayoutUnit width, LayoutUnit height)
    : m_rect(x, y, width, height)
{
}

RoundedRect::RoundedRect(const LayoutRect& rect, const Radii& radii)
    : m_rect(rect)
    , m_radii(radii)
{
}

RoundedRect::RoundedRect(const LayoutRect& rect, const LayoutSize& topLeft, const LayoutSize& topRight, const LayoutSize& bottomLeft, const LayoutSize& bottomRight)
    : m_rect(rect)
    , m_radii(topLeft, topRight, bottomLeft, bottomRight)
{
}

bool RoundedRect::isRenderable() const
{
    return m_radii.areRenderableInRect(m_rect);
}

void RoundedRect::adjustRadii()
{
    m_radii.makeRenderableInRect(m_rect);
}

bool RoundedRect::intersectsQuad(const FloatQuad& quad) const
{
    FloatRect rect(m_rect);
    if (!quad.intersectsRect(rect))
        return false;

    const LayoutSize& topLeft = m_radii.topLeft();
    if (!topLeft.isEmpty()) {
        FloatRect rect(m_rect.x(), m_rect.y(), topLeft.width(), topLeft.height());
        if (quad.intersectsRect(rect)) {
            FloatPoint center(m_rect.x() + topLeft.width(), m_rect.y() + topLeft.height());
            FloatSize size(topLeft.width(), topLeft.height());
            if (!quad.intersectsEllipse(center, size))
                return false;
        }
    }

    const LayoutSize& topRight = m_radii.topRight();
    if (!topRight.isEmpty()) {
        FloatRect rect(m_rect.maxX() - topRight.width(), m_rect.y(), topRight.width(), topRight.height());
        if (quad.intersectsRect(rect)) {
            FloatPoint center(m_rect.maxX() - topRight.width(), m_rect.y() + topRight.height());
            FloatSize size(topRight.width(), topRight.height());
            if (!quad.intersectsEllipse(center, size))
                return false;
        }
    }

    const LayoutSize& bottomLeft = m_radii.bottomLeft();
    if (!bottomLeft.isEmpty()) {
        FloatRect rect(m_rect.x(), m_rect.maxY() - bottomLeft.height(), bottomLeft.width(), bottomLeft.height());
        if (quad.intersectsRect(rect)) {
            FloatPoint center(m_rect.x() + bottomLeft.width(), m_rect.maxY() - bottomLeft.height());
            FloatSize size(bottomLeft.width(), bottomLeft.height());
            if (!quad.intersectsEllipse(center, size))
                return false;
        }
    }

    const LayoutSize& bottomRight = m_radii.bottomRight();
    if (!bottomRight.isEmpty()) {
        FloatRect rect(m_rect.maxX() - bottomRight.width(), m_rect.maxY() - bottomRight.height(), bottomRight.width(), bottomRight.height());
        if (quad.intersectsRect(rect)) {
            FloatPoint center(m_rect.maxX() - bottomRight.width(), m_rect.maxY() - bottomRight.height());
            FloatSize size(bottomRight.width(), bottomRight.height());
            if (!quad.intersectsEllipse(center, size))
                return false;
        }
    }

    return true;
}

bool RoundedRect::contains(const LayoutRect& otherRect) const
{
    if (!rect().contains(otherRect) || !isRenderable())
        return false;

    const LayoutSize& topLeft = m_radii.topLeft();
    if (!topLeft.isEmpty()) {
        FloatPoint center = { m_rect.x() + topLeft.width(), m_rect.y() + topLeft.height() };
        if (otherRect.x() <= center.x() && otherRect.y() <= center.y()) {
            if (!ellipseContainsPoint(center, topLeft, otherRect.minXMinYCorner()))
                return false;
        }
    }

    const LayoutSize& topRight = m_radii.topRight();
    if (!topRight.isEmpty()) {
        FloatPoint center = { m_rect.maxX() - topRight.width(), m_rect.y() + topRight.height() };
        if (otherRect.maxX() >= center.x() && otherRect.y() <= center.y()) {
            if (!ellipseContainsPoint(center, topRight, otherRect.maxXMinYCorner()))
                return false;
        }
    }

    const LayoutSize& bottomLeft = m_radii.bottomLeft();
    if (!bottomLeft.isEmpty()) {
        FloatPoint center = { m_rect.x() + bottomLeft.width(), m_rect.maxY() - bottomLeft.height() };
        if (otherRect.x() <= center.x() && otherRect.maxY() >= center.y()) {
            if (!ellipseContainsPoint(center, bottomLeft, otherRect.minXMaxYCorner()))
                return false;
        }
    }

    const LayoutSize& bottomRight = m_radii.bottomRight();
    if (!bottomRight.isEmpty()) {
        FloatPoint center = { m_rect.maxX() - bottomRight.width(), m_rect.maxY() - bottomRight.height() };
        if (otherRect.maxX() >= center.x() && otherRect.maxY() >= center.y()) {
            if (!ellipseContainsPoint(center, bottomRight, otherRect.maxXMaxYCorner()))
                return false;
        }
    }

    return true;
}

FloatRoundedRect RoundedRect::pixelSnappedRoundedRectForPainting(float deviceScaleFactor) const
{
    LayoutRect originalRect = rect();
    if (originalRect.isEmpty())
        return FloatRoundedRect(originalRect, radii());

    FloatRect pixelSnappedRect = snapRectToDevicePixels(originalRect, deviceScaleFactor);

    if (!isRenderable())
        return FloatRoundedRect(pixelSnappedRect, radii());

    // Snapping usually does not alter size, but when it does, we need to make sure that the final rect is still renderable by distributing the size delta proportionally.
    FloatRoundedRect::Radii adjustedRadii = radii();
    adjustedRadii.scale(pixelSnappedRect.width() / originalRect.width().toFloat(), pixelSnappedRect.height() / originalRect.height().toFloat());
    FloatRoundedRect snappedRoundedRect = FloatRoundedRect(pixelSnappedRect, adjustedRadii);
    if (!snappedRoundedRect.isRenderable()) {
        // Floating point mantissa overflow can produce a non-renderable rounded rect.
        adjustedRadii.shrink(1 / deviceScaleFactor);
        snappedRoundedRect.setRadii(adjustedRadii);
    }
    ASSERT(snappedRoundedRect.isRenderable());
    return snappedRoundedRect;
}

TextStream& operator<<(TextStream& ts, const RoundedRect& roundedRect)
{
    ts << roundedRect.rect();
    ts.dumpProperty("top-left", roundedRect.radii().topLeft());
    ts.dumpProperty("top-right", roundedRect.radii().topRight());
    ts.dumpProperty("bottom-left", roundedRect.radii().bottomLeft());
    ts.dumpProperty("bottom-right", roundedRect.radii().bottomRight());
    return ts;
}

} // namespace WebCore
