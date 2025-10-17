/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include "FETileSoftwareApplier.h"

#include "AffineTransform.h"
#include "FETile.h"
#include "Filter.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "Pattern.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FETileSoftwareApplier);

bool FETileSoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    RefPtr resultImage = result.imageBuffer();
    RefPtr inputImage = input.imageBuffer();
    if (!resultImage || !inputImage)
        return false;

    auto inputImageRect = input.absoluteImageRect();
    auto resultImageRect = result.absoluteImageRect();

    auto tileRect = input.maxEffectRect(filter);
    tileRect.scale(filter.filterScale());

    auto maxResultRect = result.maxEffectRect(filter);
    maxResultRect.scale(filter.filterScale());

    auto tileImage = ImageBuffer::create(tileRect.size(), filter.renderingMode(), RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
    if (!tileImage)
        return false;

    auto& tileImageContext = tileImage->context();
    tileImageContext.translate(-tileRect.location());
    tileImageContext.drawImageBuffer(*inputImage, inputImageRect.location());

    AffineTransform patternTransform;
    patternTransform.translate(tileRect.location() - maxResultRect.location());

    auto pattern = Pattern::create({ tileImage.releaseNonNull() }, { true, true, patternTransform });

    auto& resultContext = resultImage->context();
    resultContext.setFillPattern(WTFMove(pattern));
    resultContext.fillRect(FloatRect(FloatPoint(), resultImageRect.size()));
    return true;
}

} // namespace WebCore
