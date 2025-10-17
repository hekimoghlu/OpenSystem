/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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

#include "IntPoint.h"
#include <wtf/JSONValues.h>
#include <wtf/MathExtras.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(IOS_FAMILY)
#include <CoreGraphics/CoreGraphics.h>
#endif

#if USE(CG)
typedef struct CGSize CGSize;
#endif

#if PLATFORM(MAC)
#ifdef NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES
typedef struct CGSize NSSize;
#else
typedef struct _NSSize NSSize;
#endif
#endif // PLATFORM(MAC)

namespace WTF {
class TextStream;
}

namespace WebCore {

class IntSize;

class FloatSize {
public:
    constexpr FloatSize() = default;
    constexpr FloatSize(float width, float height) : m_width(width), m_height(height) { }
    constexpr FloatSize(const IntSize& size) : m_width(size.width()), m_height(size.height()) { }

    static FloatSize narrowPrecision(double width, double height);

    constexpr float width() const { return m_width; }
    constexpr float height() const { return m_height; }

    constexpr float minDimension() const { return std::min(m_width, m_height); }
    constexpr float maxDimension() const { return std::max(m_width, m_height); }

    void setWidth(float width) { m_width = width; }
    void setHeight(float height) { m_height = height; }

    constexpr bool isEmpty() const { return m_width <= 0 || m_height <= 0; }
    constexpr bool isZero() const;
    bool isExpressibleAsIntSize() const;

    constexpr float aspectRatio() const { return m_width / m_height; }
    constexpr double aspectRatioDouble() const { return m_width / static_cast<double>(m_height); }

    void expand(float width, float height)
    {
        m_width += width;
        m_height += height;
    }

    void scale(float s)
    {
        m_width *= s;
        m_height *= s;
    }

    void scale(float scaleX, float scaleY)
    {
        m_width *= scaleX;
        m_height *= scaleY;
    }

    constexpr FloatSize scaled(float s) const
    {
        return { m_width * s, m_height * s };
    }

    constexpr FloatSize scaled(float scaleX, float scaleY) const
    {
        return { m_width * scaleX, m_height * scaleY };
    }

    WEBCORE_EXPORT FloatSize constrainedBetween(const FloatSize& min, const FloatSize& max) const;

    constexpr FloatSize expandedTo(const FloatSize& other) const
    {
        return FloatSize(m_width > other.m_width ? m_width : other.m_width,
            m_height > other.m_height ? m_height : other.m_height);
    }

    constexpr FloatSize shrunkTo(const FloatSize& other) const
    {
       return FloatSize(m_width < other.m_width ? m_width : other.m_width,
           m_height < other.m_height ? m_height : other.m_height);
    }

    float diagonalLength() const
    {
        return std::hypot(m_width, m_height);
    }

    constexpr float diagonalLengthSquared() const
    {
        return m_width * m_width + m_height * m_height;
    }

    constexpr float area() const
    {
        return m_width * m_height;
    }

    constexpr FloatSize transposedSize() const
    {
        return FloatSize(m_height, m_width);
    }

#if USE(CG)
    WEBCORE_EXPORT explicit FloatSize(const CGSize&); // don't do this implicitly since it's lossy
    WEBCORE_EXPORT operator CGSize() const;
#endif

#if PLATFORM(MAC) && !defined(NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES)
    WEBCORE_EXPORT explicit FloatSize(const NSSize&); // don't do this implicitly since it's lossy
    operator NSSize() const;
#endif

    static constexpr FloatSize nanSize();
    constexpr bool isNaN() const;

    WEBCORE_EXPORT String toJSONString() const;
    WEBCORE_EXPORT Ref<JSON::Object> toJSONObject() const;

    friend bool operator==(const FloatSize&, const FloatSize&) = default;

    struct MarkableTraits {
        constexpr static bool isEmptyValue(const FloatSize& size)
        {
            return size.isNaN();
        }

        constexpr static FloatSize emptyValue()
        {
            return FloatSize::nanSize();
        }
    };

private:
    float m_width { 0 };
    float m_height { 0 };
};

constexpr bool FloatSize::isZero() const
{
    return fabsConstExpr(m_width) < std::numeric_limits<float>::epsilon() && fabsConstExpr(m_height) < std::numeric_limits<float>::epsilon();
}

inline FloatSize& operator+=(FloatSize& a, const FloatSize& b)
{
    a.setWidth(a.width() + b.width());
    a.setHeight(a.height() + b.height());
    return a;
}

inline FloatSize& operator-=(FloatSize& a, const FloatSize& b)
{
    a.setWidth(a.width() - b.width());
    a.setHeight(a.height() - b.height());
    return a;
}

constexpr FloatSize operator+(const FloatSize& a, const FloatSize& b)
{
    return FloatSize(a.width() + b.width(), a.height() + b.height());
}

constexpr FloatSize operator-(const FloatSize& a, const FloatSize& b)
{
    return FloatSize(a.width() - b.width(), a.height() - b.height());
}

constexpr FloatSize operator-(const FloatSize& size)
{
    return FloatSize(-size.width(), -size.height());
}

constexpr FloatSize operator*(const FloatSize& a, float b)
{
    return FloatSize(a.width() * b, a.height() * b);
}

constexpr FloatSize operator*(float a, const FloatSize& b)
{
    return FloatSize(a * b.width(), a * b.height());
}

constexpr FloatSize operator*(const FloatSize& a, const FloatSize& b)
{
    return FloatSize(a.width() * b.width(), a.height() * b.height());
}

constexpr FloatSize operator/(const FloatSize& a, const FloatSize& b)
{
    return FloatSize(a.width() / b.width(), a.height() / b.height());
}

constexpr FloatSize operator/(const FloatSize& a, float b)
{
    return FloatSize(a.width() / b, a.height() / b);
}

constexpr FloatSize operator/(float a, const FloatSize& b)
{
    return FloatSize(a / b.width(), a / b.height());
}

inline bool areEssentiallyEqual(const FloatSize& a, const FloatSize& b)
{
    return WTF::areEssentiallyEqual(a.width(), b.width()) && WTF::areEssentiallyEqual(a.height(), b.height());
}

inline IntSize roundedIntSize(const FloatSize& p)
{
    return IntSize(clampToInteger(roundf(p.width())), clampToInteger(roundf(p.height())));
}

inline IntSize flooredIntSize(const FloatSize& p)
{
    return IntSize(clampToInteger(floorf(p.width())), clampToInteger(floorf(p.height())));
}

inline IntSize expandedIntSize(const FloatSize& p)
{
    return IntSize(clampToInteger(ceilf(p.width())), clampToInteger(ceilf(p.height())));
}

inline IntPoint flooredIntPoint(const FloatSize& p)
{
    return IntPoint(clampToInteger(floorf(p.width())), clampToInteger(floorf(p.height())));
}

constexpr FloatSize FloatSize::nanSize()
{
    return {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
    };
}

constexpr bool FloatSize::isNaN() const
{
    return isNaNConstExpr(width());
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FloatSize&);

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::FloatSize>;
template<> struct HashTraits<WebCore::FloatSize>;

template<typename Type> struct LogArgument;
template <>
struct LogArgument<WebCore::FloatSize> {
    static String toString(const WebCore::FloatSize& size)
    {
        return size.toJSONString();
    }
};
    
} // namespace WTF
