/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "StyleTransformData.h"

#include "RenderStyleInlines.h"
#include "RenderStyleDifference.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleTransformData);

StyleTransformData::StyleTransformData()
    : operations(RenderStyle::initialTransform())
    , x(RenderStyle::initialTransformOriginX())
    , y(RenderStyle::initialTransformOriginY())
    , z(RenderStyle::initialTransformOriginZ())
    , transformBox(RenderStyle::initialTransformBox())
{
}

inline StyleTransformData::StyleTransformData(const StyleTransformData& other)
    : RefCounted<StyleTransformData>()
    , operations(other.operations)
    , x(other.x)
    , y(other.y)
    , z(other.z)
    , transformBox(other.transformBox)
{
}

Ref<StyleTransformData> StyleTransformData::copy() const
{
    return adoptRef(*new StyleTransformData(*this));
}

bool StyleTransformData::operator==(const StyleTransformData& other) const
{
    return x == other.x && y == other.y && z == other.z && transformBox == other.transformBox && operations == other.operations;
}

#if !LOG_DISABLED
void StyleTransformData::dumpDifferences(TextStream& ts, const StyleTransformData& other) const
{
    LOG_IF_DIFFERENT(operations);
    LOG_IF_DIFFERENT(x);
    LOG_IF_DIFFERENT(y);
    LOG_IF_DIFFERENT(z);
    LOG_IF_DIFFERENT(transformBox);
}
#endif // !LOG_DISABLED

} // namespace WebCore
