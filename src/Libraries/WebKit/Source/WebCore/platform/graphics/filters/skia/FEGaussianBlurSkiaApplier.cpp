/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "FEGaussianBlurSkiaApplier.h"

#if USE(SKIA)

#include "FEGaussianBlur.h"
#include "Filter.h"
#include "FilterImage.h"
#include "GraphicsContext.h"
#include "NativeImage.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib/Win ports
#include <skia/core/SkCanvas.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <skia/effects/SkImageFilters.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEGaussianBlurSkiaApplier);

bool FEGaussianBlurSkiaApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    ASSERT(inputs.size() == 1);
    auto& input = inputs[0].get();

    RefPtr resultImage = result.imageBuffer();
    RefPtr sourceImage = input.imageBuffer();
    if (!resultImage || !sourceImage)
        return false;

    auto nativeImage = sourceImage->createNativeImageReference();
    if (!nativeImage || !nativeImage->platformImage())
        return false;

    FloatSize sigma = FloatSize(m_effect.stdDeviationX(), m_effect.stdDeviationY()) * filter.filterScale();

    SkPaint paint;
    paint.setImageFilter(SkImageFilters::Blur(sigma.width(), sigma.height(), nullptr));

    auto inputOffsetWithinResult = input.absoluteImageRectRelativeTo(result).location();
    resultImage->context().platformContext()->drawImage(nativeImage->platformImage(), inputOffsetWithinResult.x(), inputOffsetWithinResult.y(), { }, &paint);
    return true;
}

} // namespace WebCore

#endif // USE(SKIA)
