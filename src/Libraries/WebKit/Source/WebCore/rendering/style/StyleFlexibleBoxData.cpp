/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include "StyleFlexibleBoxData.h"

#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleFlexibleBoxData);

StyleFlexibleBoxData::StyleFlexibleBoxData()
    : flexGrow(RenderStyle::initialFlexGrow())
    , flexShrink(RenderStyle::initialFlexShrink())
    , flexBasis(RenderStyle::initialFlexBasis())
    , flexDirection(static_cast<unsigned>(RenderStyle::initialFlexDirection()))
    , flexWrap(static_cast<unsigned>(RenderStyle::initialFlexWrap()))
{
}

inline StyleFlexibleBoxData::StyleFlexibleBoxData(const StyleFlexibleBoxData& other)
    : RefCounted<StyleFlexibleBoxData>()
    , flexGrow(other.flexGrow)
    , flexShrink(other.flexShrink)
    , flexBasis(other.flexBasis)
    , flexDirection(other.flexDirection)
    , flexWrap(other.flexWrap)
{
}

Ref<StyleFlexibleBoxData> StyleFlexibleBoxData::copy() const
{
    return adoptRef(*new StyleFlexibleBoxData(*this));
}

bool StyleFlexibleBoxData::operator==(const StyleFlexibleBoxData& other) const
{
    return flexGrow == other.flexGrow && flexShrink == other.flexShrink && flexBasis == other.flexBasis
        && flexDirection == other.flexDirection && flexWrap == other.flexWrap;
}

#if !LOG_DISABLED
void StyleFlexibleBoxData::dumpDifferences(TextStream& ts, const StyleFlexibleBoxData& other) const
{
    LOG_IF_DIFFERENT(flexGrow);
    LOG_IF_DIFFERENT(flexShrink);
    LOG_IF_DIFFERENT(flexBasis);

    LOG_IF_DIFFERENT_WITH_CAST(FlexDirection, flexDirection);
    LOG_IF_DIFFERENT_WITH_CAST(FlexWrap, flexWrap);
}
#endif // !LOG_DISABLED

}
