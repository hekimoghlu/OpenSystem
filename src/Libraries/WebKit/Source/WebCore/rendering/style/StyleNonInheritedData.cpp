/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include "StyleNonInheritedData.h"

#include "StyleBoxData.h"
#include "StyleBackgroundData.h"
#include "StyleSurroundData.h"
#include "StyleMiscNonInheritedData.h"
#include "StyleRareNonInheritedData.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleNonInheritedData);

StyleNonInheritedData::StyleNonInheritedData()
    : boxData(StyleBoxData::create())
    , backgroundData(StyleBackgroundData::create())
    , surroundData(StyleSurroundData::create())
    , miscData(StyleMiscNonInheritedData::create())
    , rareData(StyleRareNonInheritedData::create())
{
}

StyleNonInheritedData::StyleNonInheritedData(const StyleNonInheritedData& other)
    : RefCounted<StyleNonInheritedData>()
    , boxData(other.boxData)
    , backgroundData(other.backgroundData)
    , surroundData(other.surroundData)
    , miscData(other.miscData)
    , rareData(other.rareData)
{
    ASSERT(other == *this, "StyleNonInheritedData should be properly copied.");
}

Ref<StyleNonInheritedData> StyleNonInheritedData::create()
{
    return adoptRef(*new StyleNonInheritedData);
}

Ref<StyleNonInheritedData> StyleNonInheritedData::copy() const
{
    return adoptRef(*new StyleNonInheritedData(*this));
}

bool StyleNonInheritedData::operator==(const StyleNonInheritedData& other) const
{
    return boxData == other.boxData
        && backgroundData == other.backgroundData
        && surroundData == other.surroundData
        && miscData == other.miscData
        && rareData == other.rareData;
}

#if !LOG_DISABLED
void StyleNonInheritedData::dumpDifferences(TextStream& ts, const StyleNonInheritedData& other) const
{
    boxData->dumpDifferences(ts, *other.boxData);
    backgroundData->dumpDifferences(ts, *other.backgroundData);
    surroundData->dumpDifferences(ts, *other.surroundData);

    miscData->dumpDifferences(ts, *other.miscData);
    rareData->dumpDifferences(ts, *other.rareData);
}
#endif

} // namespace WebCore
