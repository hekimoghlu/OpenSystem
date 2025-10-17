/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "StyleGridItemData.h"

#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleGridItemData);

StyleGridItemData::StyleGridItemData()
    : gridColumnStart(RenderStyle::initialGridItemColumnStart())
    , gridColumnEnd(RenderStyle::initialGridItemColumnEnd())
    , gridRowStart(RenderStyle::initialGridItemRowStart())
    , gridRowEnd(RenderStyle::initialGridItemRowEnd())
{
}

inline StyleGridItemData::StyleGridItemData(const StyleGridItemData& o)
    : RefCounted<StyleGridItemData>()
    , gridColumnStart(o.gridColumnStart)
    , gridColumnEnd(o.gridColumnEnd)
    , gridRowStart(o.gridRowStart)
    , gridRowEnd(o.gridRowEnd)
{
}

Ref<StyleGridItemData> StyleGridItemData::copy() const
{
    return adoptRef(*new StyleGridItemData(*this));
}

#if !LOG_DISABLED
void StyleGridItemData::dumpDifferences(TextStream& ts, const StyleGridItemData& other) const
{
    LOG_IF_DIFFERENT(gridColumnStart);
    LOG_IF_DIFFERENT(gridColumnEnd);
    LOG_IF_DIFFERENT(gridRowStart);
    LOG_IF_DIFFERENT(gridRowEnd);
}
#endif

} // namespace WebCore
