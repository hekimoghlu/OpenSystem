/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#include "StyleBackgroundData.h"

#include "BorderData.h"
#include "RenderStyleConstants.h"
#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleBackgroundData);

StyleBackgroundData::StyleBackgroundData()
    : background(FillLayer::create(FillLayerType::Background))
    , color(RenderStyle::initialBackgroundColor())
{
}

inline StyleBackgroundData::StyleBackgroundData(const StyleBackgroundData& other)
    : RefCounted<StyleBackgroundData>()
    , background(other.background)
    , color(other.color)
    , outline(other.outline)
{
}

Ref<StyleBackgroundData> StyleBackgroundData::copy() const
{
    return adoptRef(*new StyleBackgroundData(*this));
}

bool StyleBackgroundData::operator==(const StyleBackgroundData& other) const
{
    return background == other.background && color == other.color && outline == other.outline;
}

bool StyleBackgroundData::isEquivalentForPainting(const StyleBackgroundData& other, bool currentColorDiffers) const
{
    if (background != other.background || color != other.color)
        return false;
    if (currentColorDiffers && color.containsCurrentColor())
        return false;
    if (!outline.isVisible() && !other.outline.isVisible())
        return true;
    if (currentColorDiffers && outline.color().containsCurrentColor())
        return false;
    return outline == other.outline;
}

void StyleBackgroundData::dump(TextStream& ts, DumpStyleValues behavior) const
{
    if (behavior == DumpStyleValues::All || *background != FillLayer::create(FillLayerType::Background).get())
        ts.dumpProperty("background-image", background);
    if (behavior == DumpStyleValues::All || color != RenderStyle::initialBackgroundColor())
        ts.dumpProperty("background-color", color);
    if (behavior == DumpStyleValues::All || outline != OutlineValue())
        ts.dumpProperty("outline", outline);
}

#if !LOG_DISABLED
void StyleBackgroundData::dumpDifferences(TextStream& ts, const StyleBackgroundData& other) const
{
    LOG_IF_DIFFERENT(background);
    LOG_IF_DIFFERENT(color);
    LOG_IF_DIFFERENT(outline);
}
#endif

TextStream& operator<<(TextStream& ts, const StyleBackgroundData& backgroundData)
{
    backgroundData.dump(ts);
    return ts;
}

} // namespace WebCore
