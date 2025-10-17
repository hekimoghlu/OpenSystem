/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "StyleMiscNonInheritedData.h"

#include "AnimationList.h"
#include "ContentData.h"
#include "FillLayer.h"
#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"
#include "ShadowData.h"
#include "StyleDeprecatedFlexibleBoxData.h"
#include "StyleFilterData.h"
#include "StyleFlexibleBoxData.h"
#include "StyleMultiColData.h"
#include "StyleTransformData.h"
#include "StyleVisitedLinkColorData.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleMiscNonInheritedData);

StyleMiscNonInheritedData::StyleMiscNonInheritedData()
    : opacity(RenderStyle::initialOpacity())
    , deprecatedFlexibleBox(StyleDeprecatedFlexibleBoxData::create())
    , flexibleBox(StyleFlexibleBoxData::create())
    , multiCol(StyleMultiColData::create())
    , filter(StyleFilterData::create())
    , transform(StyleTransformData::create())
    , mask(FillLayer::create(FillLayerType::Mask))
    , visitedLinkColor(StyleVisitedLinkColorData::create())
    , aspectRatioWidth(RenderStyle::initialAspectRatioWidth())
    , aspectRatioHeight(RenderStyle::initialAspectRatioHeight())
    , alignContent(RenderStyle::initialContentAlignment())
    , justifyContent(RenderStyle::initialContentAlignment())
    , alignItems(RenderStyle::initialDefaultAlignment())
    , alignSelf(RenderStyle::initialSelfAlignment())
    , justifyItems(RenderStyle::initialJustifyItems())
    , justifySelf(RenderStyle::initialSelfAlignment())
    , objectPosition(RenderStyle::initialObjectPosition())
    , order(RenderStyle::initialOrder())
    , tableLayout(static_cast<unsigned>(RenderStyle::initialTableLayout()))
    , aspectRatioType(static_cast<unsigned>(RenderStyle::initialAspectRatioType()))
    , appearance(static_cast<unsigned>(RenderStyle::initialAppearance()))
    , usedAppearance(static_cast<unsigned>(RenderStyle::initialAppearance()))
    , textOverflow(static_cast<unsigned>(RenderStyle::initialTextOverflow()))
    , userDrag(static_cast<unsigned>(RenderStyle::initialUserDrag()))
    , objectFit(static_cast<unsigned>(RenderStyle::initialObjectFit()))
    , resize(static_cast<unsigned>(RenderStyle::initialResize()))
{
}

StyleMiscNonInheritedData::StyleMiscNonInheritedData(const StyleMiscNonInheritedData& o)
    : RefCounted<StyleMiscNonInheritedData>()
    , opacity(o.opacity)
    , deprecatedFlexibleBox(o.deprecatedFlexibleBox)
    , flexibleBox(o.flexibleBox)
    , multiCol(o.multiCol)
    , filter(o.filter)
    , transform(o.transform)
    , mask(o.mask)
    , visitedLinkColor(o.visitedLinkColor)
    , animations(o.animations ? o.animations->copy() : o.animations)
    , transitions(o.transitions ? o.transitions->copy() : o.transitions)
    , content(o.content ? o.content->clone() : nullptr)
    , boxShadow(o.boxShadow ? makeUnique<ShadowData>(*o.boxShadow) : nullptr)
    , altText(o.altText)
    , aspectRatioWidth(o.aspectRatioWidth)
    , aspectRatioHeight(o.aspectRatioHeight)
    , alignContent(o.alignContent)
    , justifyContent(o.justifyContent)
    , alignItems(o.alignItems)
    , alignSelf(o.alignSelf)
    , justifyItems(o.justifyItems)
    , justifySelf(o.justifySelf)
    , objectPosition(o.objectPosition)
    , order(o.order)
    , hasAttrContent(o.hasAttrContent)
    , hasDisplayAffectedByAnimations(o.hasDisplayAffectedByAnimations)
#if ENABLE(DARK_MODE_CSS)
    , hasExplicitlySetColorScheme(o.hasExplicitlySetColorScheme)
#endif
    , hasExplicitlySetDirection(o.hasExplicitlySetDirection)
    , hasExplicitlySetWritingMode(o.hasExplicitlySetWritingMode)
    , tableLayout(o.tableLayout)
    , aspectRatioType(o.aspectRatioType)
    , appearance(o.appearance)
    , usedAppearance(o.usedAppearance)
    , textOverflow(o.textOverflow)
    , userDrag(o.userDrag)
    , objectFit(o.objectFit)
    , resize(o.resize)
{
}

StyleMiscNonInheritedData::~StyleMiscNonInheritedData() = default;

Ref<StyleMiscNonInheritedData> StyleMiscNonInheritedData::copy() const
{
    return adoptRef(*new StyleMiscNonInheritedData(*this));
}

bool StyleMiscNonInheritedData::operator==(const StyleMiscNonInheritedData& o) const
{
    return opacity == o.opacity
        && deprecatedFlexibleBox == o.deprecatedFlexibleBox
        && flexibleBox == o.flexibleBox
        && multiCol == o.multiCol
        && filter == o.filter
        && transform == o.transform
        && mask == o.mask
        && visitedLinkColor == o.visitedLinkColor
        && arePointingToEqualData(animations, o.animations)
        && arePointingToEqualData(transitions, o.transitions)
        && contentDataEquivalent(o)
        && arePointingToEqualData(boxShadow, o.boxShadow)
        && altText == o.altText
        && aspectRatioWidth == o.aspectRatioWidth
        && aspectRatioHeight == o.aspectRatioHeight
        && alignContent == o.alignContent
        && justifyContent == o.justifyContent
        && alignItems == o.alignItems
        && alignSelf == o.alignSelf
        && justifyItems == o.justifyItems
        && justifySelf == o.justifySelf
        && objectPosition == o.objectPosition
        && order == o.order
        && hasAttrContent == o.hasAttrContent
        && hasDisplayAffectedByAnimations == o.hasDisplayAffectedByAnimations
#if ENABLE(DARK_MODE_CSS)
        && hasExplicitlySetColorScheme == o.hasExplicitlySetColorScheme
#endif
        && hasExplicitlySetDirection == o.hasExplicitlySetDirection
        && hasExplicitlySetWritingMode == o.hasExplicitlySetWritingMode
        && aspectRatioType == o.aspectRatioType
        && tableLayout == o.tableLayout
        && appearance == o.appearance
        && usedAppearance == o.usedAppearance
        && textOverflow == o.textOverflow
        && userDrag == o.userDrag
        && objectFit == o.objectFit
        && resize == o.resize;
}

bool StyleMiscNonInheritedData::contentDataEquivalent(const StyleMiscNonInheritedData& other) const
{
    auto* a = content.get();
    auto* b = other.content.get();
    while (a && b && *a == *b) {
        a = a->next();
        b = b->next();
    }
    return !a && !b;
}

bool StyleMiscNonInheritedData::hasFilters() const
{
    return !filter->operations.isEmpty();
}

#if !LOG_DISABLED
void StyleMiscNonInheritedData::dumpDifferences(TextStream& ts, const StyleMiscNonInheritedData& other) const
{
    LOG_IF_DIFFERENT(opacity);

    deprecatedFlexibleBox->dumpDifferences(ts, other.deprecatedFlexibleBox);
    flexibleBox->dumpDifferences(ts, other.flexibleBox);
    multiCol->dumpDifferences(ts, other.multiCol);

    filter->dumpDifferences(ts, other.filter);
    transform->dumpDifferences(ts, other.transform);

    LOG_IF_DIFFERENT(mask);

    visitedLinkColor->dumpDifferences(ts, other.visitedLinkColor);

    LOG_IF_DIFFERENT(animations);
    LOG_IF_DIFFERENT(transitions);

    LOG_IF_DIFFERENT(content);
    LOG_IF_DIFFERENT(boxShadow);

    LOG_IF_DIFFERENT(altText);
    LOG_IF_DIFFERENT(aspectRatioWidth);
    LOG_IF_DIFFERENT(aspectRatioHeight);

    LOG_IF_DIFFERENT(alignContent);
    LOG_IF_DIFFERENT(justifyContent);
    LOG_IF_DIFFERENT(alignItems);
    LOG_IF_DIFFERENT(alignSelf);
    LOG_IF_DIFFERENT(justifyItems);
    LOG_IF_DIFFERENT(justifySelf);
    LOG_IF_DIFFERENT(objectPosition);
    LOG_IF_DIFFERENT(order);

    LOG_IF_DIFFERENT_WITH_CAST(bool, hasAttrContent);
    LOG_IF_DIFFERENT_WITH_CAST(bool, hasDisplayAffectedByAnimations);

#if ENABLE(DARK_MODE_CSS)
    LOG_IF_DIFFERENT_WITH_CAST(bool, hasExplicitlySetColorScheme);
#endif

    LOG_IF_DIFFERENT_WITH_CAST(bool, hasExplicitlySetDirection);
    LOG_IF_DIFFERENT_WITH_CAST(bool, hasExplicitlySetWritingMode);

    LOG_IF_DIFFERENT_WITH_CAST(TableLayoutType, tableLayout);
    LOG_IF_DIFFERENT_WITH_CAST(AspectRatioType, aspectRatioType);
    LOG_IF_DIFFERENT_WITH_CAST(StyleAppearance, appearance);
    LOG_IF_DIFFERENT_WITH_CAST(StyleAppearance, usedAppearance);

    LOG_IF_DIFFERENT_WITH_CAST(bool, textOverflow);

    LOG_IF_DIFFERENT_WITH_CAST(UserDrag, objectFit);
    LOG_IF_DIFFERENT_WITH_CAST(ObjectFit, textOverflow);
    LOG_IF_DIFFERENT_WITH_CAST(Resize, resize);
}
#endif // !LOG_DISABLED

} // namespace WebCore
