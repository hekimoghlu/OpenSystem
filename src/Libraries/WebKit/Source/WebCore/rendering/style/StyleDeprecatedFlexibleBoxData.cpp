/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#include "StyleDeprecatedFlexibleBoxData.h"

#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleDeprecatedFlexibleBoxData);

StyleDeprecatedFlexibleBoxData::StyleDeprecatedFlexibleBoxData()
    : flex(RenderStyle::initialBoxFlex())
    , flexGroup(RenderStyle::initialBoxFlexGroup())
    , ordinalGroup(RenderStyle::initialBoxOrdinalGroup())
    , align(static_cast<unsigned>(RenderStyle::initialBoxAlign()))
    , pack(static_cast<unsigned>(RenderStyle::initialBoxPack()))
    , orient(static_cast<unsigned>(RenderStyle::initialBoxOrient()))
    , lines(static_cast<unsigned>(RenderStyle::initialBoxLines()))
{
}

inline StyleDeprecatedFlexibleBoxData::StyleDeprecatedFlexibleBoxData(const StyleDeprecatedFlexibleBoxData& other)
    : RefCounted<StyleDeprecatedFlexibleBoxData>()
    , flex(other.flex)
    , flexGroup(other.flexGroup)
    , ordinalGroup(other.ordinalGroup)
    , align(other.align)
    , pack(other.pack)
    , orient(other.orient)
    , lines(other.lines)
{
}

Ref<StyleDeprecatedFlexibleBoxData> StyleDeprecatedFlexibleBoxData::copy() const
{
    return adoptRef(*new StyleDeprecatedFlexibleBoxData(*this));
}

bool StyleDeprecatedFlexibleBoxData::operator==(const StyleDeprecatedFlexibleBoxData& other) const
{
    return flex == other.flex && flexGroup == other.flexGroup
        && ordinalGroup == other.ordinalGroup && align == other.align
        && pack == other.pack && orient == other.orient && lines == other.lines;
}

#if !LOG_DISABLED
void StyleDeprecatedFlexibleBoxData::dumpDifferences(TextStream& ts, const StyleDeprecatedFlexibleBoxData& other) const
{
    LOG_IF_DIFFERENT(flex);
    LOG_IF_DIFFERENT(flexGroup);
    LOG_IF_DIFFERENT(ordinalGroup);

    LOG_IF_DIFFERENT_WITH_CAST(BoxAlignment, align);
    LOG_IF_DIFFERENT_WITH_CAST(BoxPack, pack);
    LOG_IF_DIFFERENT_WITH_CAST(BoxOrient, orient);
    LOG_IF_DIFFERENT_WITH_CAST(BoxLines, lines);
}
#endif // !LOG_DISABLED

} // namespace WebCore
