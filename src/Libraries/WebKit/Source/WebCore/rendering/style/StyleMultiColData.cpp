/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "StyleMultiColData.h"

#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleMultiColData);

StyleMultiColData::StyleMultiColData()
    : count(RenderStyle::initialColumnCount())
    , autoWidth(true)
    , autoCount(true)
    , fill(static_cast<unsigned>(RenderStyle::initialColumnFill()))
    , columnSpan(false)
    , axis(static_cast<unsigned>(RenderStyle::initialColumnAxis()))
    , progression(static_cast<unsigned>(RenderStyle::initialColumnProgression()))
{
}

inline StyleMultiColData::StyleMultiColData(const StyleMultiColData& other)
    : RefCounted<StyleMultiColData>()
    , width(other.width)
    , count(other.count)
    , rule(other.rule)
    , visitedLinkColumnRuleColor(other.visitedLinkColumnRuleColor)
    , autoWidth(other.autoWidth)
    , autoCount(other.autoCount)
    , fill(other.fill)
    , columnSpan(other.columnSpan)
    , axis(other.axis)
    , progression(other.progression)
{
}

Ref<StyleMultiColData> StyleMultiColData::copy() const
{
    return adoptRef(*new StyleMultiColData(*this));
}

bool StyleMultiColData::operator==(const StyleMultiColData& other) const
{
    return width == other.width && count == other.count
        && rule == other.rule && visitedLinkColumnRuleColor == other.visitedLinkColumnRuleColor
        && autoWidth == other.autoWidth && autoCount == other.autoCount
        && fill == other.fill && columnSpan == other.columnSpan
        && axis == other.axis && progression == other.progression;
}

#if !LOG_DISABLED
void StyleMultiColData::dumpDifferences(TextStream& ts, const StyleMultiColData& other) const
{
    LOG_IF_DIFFERENT(width);
    LOG_IF_DIFFERENT(count);
    LOG_IF_DIFFERENT(rule);
    LOG_IF_DIFFERENT(visitedLinkColumnRuleColor);

    LOG_IF_DIFFERENT(autoWidth);
    LOG_IF_DIFFERENT(autoCount);

    LOG_IF_DIFFERENT_WITH_CAST(ColumnFill, fill);
    LOG_IF_DIFFERENT_WITH_CAST(ColumnSpan, columnSpan);
    LOG_IF_DIFFERENT_WITH_CAST(ColumnAxis, axis);
    LOG_IF_DIFFERENT_WITH_CAST(ColumnProgression, progression);
}
#endif // !LOG_DISABLED

} // namespace WebCore
