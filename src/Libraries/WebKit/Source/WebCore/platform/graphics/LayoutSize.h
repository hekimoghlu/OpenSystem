/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "FloatSize.h"
#include "IntSize.h"
#include "LayoutUnit.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

class LayoutPoint;

enum AspectRatioFit {
    AspectRatioFitShrink,
    AspectRatioFitGrow
};

class LayoutSize {
public:
    LayoutSize() = default;
    LayoutSize(const IntSize& size) : m_width(size.width()), m_height(size.height()) { }
    template<typename T, typename U> LayoutSize(T width, U height) : m_width(width), m_height(height) { }

    explicit LayoutSize(const FloatSize& size) : m_width(size.width()), m_height(size.height()) { }
    
    LayoutUnit width() const { return m_width; }
    LayoutUnit height() const { return m_height; }

    LayoutUnit minDimension() const { return std::min(m_width, m_height); }
    LayoutUnit maxDimension() const { return std::max(m_width, m_height); }

    template<typename T> void setWidth(T width) { m_width = width; }
    template<typename T> void setHeight(T height) { m_height = height; }

    bool isEmpty() const { return m_width.rawValue() <= 0 || m_height.rawValue() <= 0; }
    bool isZero() const { return !m_width && !m_height; }

    float aspectRatio() const { return static_cast<float>(m_width) / static_cast<float>(m_height); }
    
    template<typename T, typename U>
    void expand(T width, U height)
    {
        m_width += width;
        m_height += height;
    }
    
    void shrink(LayoutUnit width, LayoutUnit height)
    {
        m_width -= width;
        m_height -= height;
    }
    
    void scale(float scale)
    {
        m_width *= scale;
        m_height *= scale;
    }

    void scale(float widthScale, float heightScale)
    {
        m_width *= widthScale;
        m_height *= heightScale;
    }

    LayoutSize constrainedBetween(const LayoutSize& min, const LayoutSize& max) const;
    
    LayoutSize expandedTo(const LayoutSize& other) const
    {
        return LayoutSize(m_width > other.m_width ? m_width : other.m_width,
            m_height > other.m_height ? m_height : other.m_height);
    }

    LayoutSize shrunkTo(const LayoutSize& other) const
    {
        return LayoutSize(m_width < other.m_width ? m_width : other.m_width,
            m_height < other.m_height ? m_height : other.m_height);
    }

    void clampNegativeToZero()
    {
        *this = expandedTo(LayoutSize());
    }
    
    void clampToMinimumSize(const LayoutSize& minimumSize)
    {
        if (m_width < minimumSize.width())
            m_width = minimumSize.width();
        if (m_height < minimumSize.height())
            m_height = minimumSize.height();
    }

    LayoutSize transposedSize() const
    {
        return LayoutSize(m_height, m_width);
    }

    operator FloatSize() const { return FloatSize(m_width, m_height); }

    LayoutSize fitToAspectRatio(const LayoutSize& aspectRatio, AspectRatioFit fit) const
    {
        float heightScale = height().toFloat() / aspectRatio.height().toFloat();
        float widthScale = width().toFloat() / aspectRatio.width().toFloat();

        if ((widthScale > heightScale) != (fit == AspectRatioFitGrow))
            return LayoutSize(height() * aspectRatio.width() / aspectRatio.height(), height());

        return LayoutSize(width(), width() * aspectRatio.height() / aspectRatio.width());
    }

    bool mightBeSaturated() const
    {
        return m_width.mightBeSaturated() || m_height.mightBeSaturated();
    }

    friend bool operator==(const LayoutSize&, const LayoutSize&) = default;
private:
    LayoutUnit m_width;
    LayoutUnit m_height;
};

inline LayoutSize& operator+=(LayoutSize& a, const LayoutSize& b)
{
    a.setWidth(a.width() + b.width());
    a.setHeight(a.height() + b.height());
    return a;
}

inline LayoutSize& operator-=(LayoutSize& a, const LayoutSize& b)
{
    a.setWidth(a.width() - b.width());
    a.setHeight(a.height() - b.height());
    return a;
}

inline LayoutSize operator+(const LayoutSize& a, const LayoutSize& b)
{
    return LayoutSize(a.width() + b.width(), a.height() + b.height());
}

inline LayoutSize operator-(const LayoutSize& a, const LayoutSize& b)
{
    return LayoutSize(a.width() - b.width(), a.height() - b.height());
}

inline LayoutSize operator-(const LayoutSize& size)
{
    return LayoutSize(-size.width(), -size.height());
}

inline IntSize flooredIntSize(const LayoutSize& s)
{
    return IntSize(s.width().floor(), s.height().floor());
}

inline IntSize ceiledIntSize(const LayoutSize& s)
{
    return IntSize(s.width().ceil(), s.height().ceil());
}

inline IntSize roundedIntSize(const LayoutSize& s)
{
    return IntSize(s.width().round(), s.height().round());
}

inline FloatSize floorSizeToDevicePixels(const LayoutSize& size, float pixelSnappingFactor)
{
    return FloatSize(floorToDevicePixel(size.width(), pixelSnappingFactor), floorToDevicePixel(size.height(), pixelSnappingFactor));
}

inline FloatSize roundSizeToDevicePixels(const LayoutSize& size, float pixelSnappingFactor)
{
    return FloatSize(roundToDevicePixel(size.width(), pixelSnappingFactor), roundToDevicePixel(size.height(), pixelSnappingFactor));
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const LayoutSize&);

} // namespace WebCore

