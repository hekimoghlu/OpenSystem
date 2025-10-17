/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#include "FEBlendSoftwareApplier.h"

#if !HAVE(ARM_NEON_INTRINSICS)

#include "FEBlend.h"
#include "FloatPoint.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEBlendSoftwareApplier);

bool FEBlendSoftwareApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();
    auto& input2 = inputs[1].get();

    RefPtr resultImage = result.imageBuffer();
    if (!resultImage)
        return false;

    RefPtr inputImage = input.imageBuffer();
    RefPtr inputImage2 = input2.imageBuffer();
    if (!inputImage || !inputImage2)
        return false;

    auto& filterContext = resultImage->context();
    auto inputImageRect = input.absoluteImageRectRelativeTo(result);
    auto inputImageRect2 = input2.absoluteImageRectRelativeTo(result);

    filterContext.drawImageBuffer(*inputImage2, inputImageRect2);
    filterContext.drawImageBuffer(*inputImage, inputImageRect, { { }, inputImage->logicalSize() }, { CompositeOperator::SourceOver, m_effect.blendMode() });
    return true;
}

} // namespace WebCore

#endif // !HAVE(ARM_NEON_INTRINSICS)
