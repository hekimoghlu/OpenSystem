/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include "FloatPoint.h"
#include "FloatSize.h"
#include "Glyph.h"
#include <wtf/Vector.h>

#if USE(CG)
#include <CoreGraphics/CGFont.h>
#include <CoreGraphics/CGGeometry.h>
#endif

namespace WebCore {

// The CG ports use the CG types directly, so an array of these types can be fed directly into CTFontShapeGlyphs().
// When you write code that interacts with these types, use the functions below!
#if USE(CG)
using GlyphBufferGlyph = CGGlyph;
using GlyphBufferAdvance = CGSize;
using GlyphBufferOrigin = CGPoint;
using GlyphBufferStringOffset = CFIndex;
#else
using GlyphBufferGlyph = Glyph;
using GlyphBufferAdvance = FloatSize;
using GlyphBufferOrigin = FloatPoint;
using GlyphBufferStringOffset = unsigned;
#endif

inline GlyphBufferAdvance makeGlyphBufferAdvance(const FloatSize&);
inline GlyphBufferAdvance makeGlyphBufferAdvance(float = 0, float = 0);
inline FloatSize size(const GlyphBufferAdvance&);
inline void setWidth(GlyphBufferAdvance&, float);
inline void setHeight(GlyphBufferAdvance&, float);
inline float width(const GlyphBufferAdvance&);
inline float height(const GlyphBufferAdvance&);
inline GlyphBufferOrigin makeGlyphBufferOrigin(const FloatPoint&);
inline GlyphBufferOrigin makeGlyphBufferOrigin(float = 0, float = 0);
inline FloatPoint point(const GlyphBufferOrigin&);
inline void setX(GlyphBufferOrigin&, float);
inline void setY(GlyphBufferOrigin&, float);
inline float x(const GlyphBufferOrigin&);
inline float y(const GlyphBufferOrigin&);

#if USE(CG)

inline GlyphBufferAdvance makeGlyphBufferAdvance(const FloatSize& size)
{
    return CGSizeMake(size.width(), size.height());
}

inline GlyphBufferAdvance makeGlyphBufferAdvance(float width, float height)
{
    return CGSizeMake(width, height);
}

inline FloatSize size(const GlyphBufferAdvance& advance)
{
    return FloatSize(advance.width, advance.height);
}

inline void setWidth(GlyphBufferAdvance& advance, float width)
{
    advance.width = width;
}

inline void setHeight(GlyphBufferAdvance& advance, float height)
{
    advance.height = height;
}

inline float width(const GlyphBufferAdvance& advance)
{
    return advance.width;
}

inline float height(const GlyphBufferAdvance& advance)
{
    return advance.height;
}

inline GlyphBufferOrigin makeGlyphBufferOrigin(const FloatPoint& point)
{
    return CGPointMake(point.x(), point.y());
}

inline GlyphBufferOrigin makeGlyphBufferOrigin(float x, float y)
{
    return CGPointMake(x, y);
}

inline FloatPoint point(const GlyphBufferOrigin& origin)
{
    return FloatPoint(origin.x, origin.y);
}

inline void setX(GlyphBufferOrigin& origin, float x)
{
    origin.x = x;
}

inline void setY(GlyphBufferOrigin& origin, float y)
{
    origin.y = y;
}

inline float x(const GlyphBufferOrigin& origin)
{
    return origin.x;
}

inline float y(const GlyphBufferOrigin& origin)
{
    return origin.y;
}

#else

inline GlyphBufferAdvance makeGlyphBufferAdvance(const FloatSize& size)
{
    return size;
}

inline GlyphBufferAdvance makeGlyphBufferAdvance(float width, float height)
{
    return FloatSize(width, height);
}

inline FloatSize size(const GlyphBufferAdvance& advance)
{
    return advance;
}

inline void setWidth(GlyphBufferAdvance& advance, float width)
{
    advance.setWidth(width);
}

inline void setHeight(GlyphBufferAdvance& advance, float height)
{
    advance.setHeight(height);
}

inline float width(const GlyphBufferAdvance& advance)
{
    return advance.width();
}

inline float height(const GlyphBufferAdvance& advance)
{
    return advance.height();
}

inline GlyphBufferOrigin makeGlyphBufferOrigin(const FloatPoint& point)
{
    return point;
}

inline GlyphBufferOrigin makeGlyphBufferOrigin(float x, float y)
{
    return FloatPoint(x, y);
}

inline FloatPoint point(const GlyphBufferOrigin& origin)
{
    return origin;
}

inline void setX(GlyphBufferOrigin& origin, float x)
{
    origin.setX(x);
}

inline void setY(GlyphBufferOrigin& origin, float y)
{
    origin.setY(y);
}

inline float x(const GlyphBufferOrigin& origin)
{
    return origin.x();
}

inline float y(const GlyphBufferOrigin& origin)
{
    return origin.y();
}

#endif // #if USE(CG)

}
