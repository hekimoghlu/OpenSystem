/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include "SourceAlphaSoftwareApplier.h"

#include "Color.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "SourceAlpha.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SourceAlphaSoftwareApplier);

bool SourceAlphaSoftwareApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    RefPtr resultImage = result.imageBuffer();
    if (!resultImage)
        return false;
    
    RefPtr inputImage = input.imageBuffer();
    if (!inputImage)
        return false;

    FloatRect imageRect(FloatPoint(), result.absoluteImageRect().size());
    auto& filterContext = resultImage->context();

    filterContext.fillRect(imageRect, Color::black);
    filterContext.drawImageBuffer(*inputImage, IntPoint(), { CompositeOperator::DestinationIn });
    return true;
}

} // namespace WebCore
