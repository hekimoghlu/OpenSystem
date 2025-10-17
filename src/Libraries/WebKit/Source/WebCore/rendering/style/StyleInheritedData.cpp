/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#include "StyleInheritedData.h"

#include "RenderStyleInlines.h"
#include "RenderStyleDifference.h"
#include "StyleFontData.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleInheritedData);

StyleInheritedData::StyleInheritedData()
    : horizontalBorderSpacing(RenderStyle::initialHorizontalBorderSpacing())
    , verticalBorderSpacing(RenderStyle::initialVerticalBorderSpacing())
    , lineHeight(RenderStyle::initialLineHeight())
#if ENABLE(TEXT_AUTOSIZING)
    , specifiedLineHeight(RenderStyle::initialLineHeight())
#endif
    , fontData(StyleFontData::create())
    , color(RenderStyle::initialColor())
    , visitedLinkColor(RenderStyle::initialColor())
{
}

inline StyleInheritedData::StyleInheritedData(const StyleInheritedData& o)
    : RefCounted<StyleInheritedData>()
    , horizontalBorderSpacing(o.horizontalBorderSpacing)
    , verticalBorderSpacing(o.verticalBorderSpacing)
    , lineHeight(o.lineHeight)
#if ENABLE(TEXT_AUTOSIZING)
    , specifiedLineHeight(o.specifiedLineHeight)
#endif
    , fontData(o.fontData)
    , color(o.color)
    , visitedLinkColor(o.visitedLinkColor)
{
    ASSERT(o == *this, "StyleInheritedData should be properly copied.");
}

Ref<StyleInheritedData> StyleInheritedData::copy() const
{
    return adoptRef(*new StyleInheritedData(*this));
}

bool StyleInheritedData::operator==(const StyleInheritedData& other) const
{
    return fastPathInheritedEqual(other) && nonFastPathInheritedEqual(other);
}

bool StyleInheritedData::fastPathInheritedEqual(const StyleInheritedData& other) const
{
    // These properties also need to have "fast-path-inherited" codegen property set.
    // Cases where other properties depend on these values need to disallow the fast path (via RenderStyle::setDisallowsFastPathInheritance).
    return color == other.color
        && visitedLinkColor == other.visitedLinkColor;
}

bool StyleInheritedData::nonFastPathInheritedEqual(const StyleInheritedData& other) const
{
    return lineHeight == other.lineHeight
#if ENABLE(TEXT_AUTOSIZING)
        && specifiedLineHeight == other.specifiedLineHeight
#endif
        && fontData == other.fontData
        && horizontalBorderSpacing == other.horizontalBorderSpacing
        && verticalBorderSpacing == other.verticalBorderSpacing;
}

void StyleInheritedData::fastPathInheritFrom(const StyleInheritedData& inheritParent)
{
    color = inheritParent.color;
    visitedLinkColor = inheritParent.visitedLinkColor;
}

#if !LOG_DISABLED
void StyleInheritedData::dumpDifferences(TextStream& ts, const StyleInheritedData& other) const
{
    fontData->dumpDifferences(ts, *other.fontData);

    LOG_IF_DIFFERENT(horizontalBorderSpacing);
    LOG_IF_DIFFERENT(verticalBorderSpacing);
    LOG_IF_DIFFERENT(lineHeight);

#if ENABLE(TEXT_AUTOSIZING)
    LOG_IF_DIFFERENT(specifiedLineHeight);
#endif

    LOG_IF_DIFFERENT(color);
    LOG_IF_DIFFERENT(visitedLinkColor);
}
#endif

} // namespace WebCore
