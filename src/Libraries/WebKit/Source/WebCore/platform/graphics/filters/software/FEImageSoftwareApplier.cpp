/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include "FEImageSoftwareApplier.h"

#include "FEImage.h"
#include "Filter.h"
#include "GraphicsContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEImageSoftwareApplier);

bool FEImageSoftwareApplier::apply(const Filter& filter, const FilterImageVector&, FilterImage& result) const
{
    RefPtr resultImage = result.imageBuffer();
    if (!resultImage)
        return false;

    auto& sourceImage = m_effect.sourceImage();
    auto primitiveSubregion = result.primitiveSubregion();
    auto& context = resultImage->context();

    if (auto nativeImage = sourceImage.nativeImageIfExists()) {
        auto imageRect = primitiveSubregion;
        auto srcRect = m_effect.sourceImageRect();
        m_effect.preserveAspectRatio().transformRect(imageRect, srcRect);
        imageRect.scale(filter.filterScale());
        imageRect = IntRect(imageRect) - result.absoluteImageRect().location();
        context.drawNativeImage(*nativeImage, imageRect, srcRect);
        return true;
    }

    if (auto imageBuffer = sourceImage.imageBufferIfExists()) {
        auto imageRect = primitiveSubregion;
        imageRect.moveBy(m_effect.sourceImageRect().location());
        imageRect.scale(filter.filterScale());
        imageRect = IntRect(imageRect) - result.absoluteImageRect().location();
        context.drawImageBuffer(*imageBuffer, imageRect.location());
        return true;
    }
    
    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore
