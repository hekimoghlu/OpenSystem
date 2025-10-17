/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "StyleSurroundData.h"

#include "RenderStyleDifference.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleSurroundData);

StyleSurroundData::StyleSurroundData()
    : hasExplicitlySetBorderBottomLeftRadius(false)
    , hasExplicitlySetBorderBottomRightRadius(false)
    , hasExplicitlySetBorderTopLeftRadius(false)
    , hasExplicitlySetBorderTopRightRadius(false)
    , margin(LengthType::Fixed)
    , padding(LengthType::Fixed)
{
}

inline StyleSurroundData::StyleSurroundData(const StyleSurroundData& o)
    : RefCounted<StyleSurroundData>()
    , hasExplicitlySetBorderBottomLeftRadius(o.hasExplicitlySetBorderBottomLeftRadius)
    , hasExplicitlySetBorderBottomRightRadius(o.hasExplicitlySetBorderBottomRightRadius)
    , hasExplicitlySetBorderTopLeftRadius(o.hasExplicitlySetBorderTopLeftRadius)
    , hasExplicitlySetBorderTopRightRadius(o.hasExplicitlySetBorderTopRightRadius)
    , offset(o.offset)
    , margin(o.margin)
    , padding(o.padding)
    , border(o.border)
{
}

Ref<StyleSurroundData> StyleSurroundData::copy() const
{
    return adoptRef(*new StyleSurroundData(*this));
}

bool StyleSurroundData::operator==(const StyleSurroundData& o) const
{
    return offset == o.offset && margin == o.margin && padding == o.padding && border == o.border
        && hasExplicitlySetBorderBottomLeftRadius == o.hasExplicitlySetBorderBottomLeftRadius
        && hasExplicitlySetBorderBottomRightRadius == o.hasExplicitlySetBorderBottomRightRadius
        && hasExplicitlySetBorderTopLeftRadius == o.hasExplicitlySetBorderTopLeftRadius
        && hasExplicitlySetBorderTopRightRadius == o.hasExplicitlySetBorderTopRightRadius;
}

#if !LOG_DISABLED
void StyleSurroundData::dumpDifferences(TextStream& ts, const StyleSurroundData& other) const
{
    LOG_IF_DIFFERENT(hasExplicitlySetBorderBottomLeftRadius);
    LOG_IF_DIFFERENT(hasExplicitlySetBorderBottomRightRadius);
    LOG_IF_DIFFERENT(hasExplicitlySetBorderTopLeftRadius);
    LOG_IF_DIFFERENT(hasExplicitlySetBorderTopRightRadius);

    LOG_IF_DIFFERENT(offset);
    LOG_IF_DIFFERENT(margin);
    LOG_IF_DIFFERENT(padding);
    LOG_IF_DIFFERENT(border);
}
#endif

} // namespace WebCore
