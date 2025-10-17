/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#include "FEOffsetSoftwareApplier.h"

#include "FEOffset.h"
#include "Filter.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEOffsetSoftwareApplier);

bool FEOffsetSoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    RefPtr resultImage = result.imageBuffer();
    RefPtr inputImage = input.imageBuffer();
    if (!resultImage || !inputImage)
        return false;

    FloatRect inputImageRect = input.absoluteImageRectRelativeTo(result);

    auto offset = filter.resolvedSize({ m_effect.dx(), m_effect.dy() });
    auto absoluteOffset = filter.scaledByFilterScale(offset);

    inputImageRect.move(absoluteOffset);

    resultImage->context().drawImageBuffer(*inputImage, inputImageRect);
    return true;
}

} // namespace WebCore
