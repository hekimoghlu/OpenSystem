/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "FECompositeSoftwareApplier.h"

#include "FEComposite.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PixelBuffer.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FECompositeSoftwareApplier);

FECompositeSoftwareApplier::FECompositeSoftwareApplier(const FEComposite& effect)
    : Base(effect)
{
    ASSERT(m_effect.operation() != CompositeOperationType::FECOMPOSITE_OPERATOR_ARITHMETIC);
}

bool FECompositeSoftwareApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
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

    switch (m_effect.operation()) {
    case CompositeOperationType::FECOMPOSITE_OPERATOR_UNKNOWN:
        return false;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_OVER:
        filterContext.drawImageBuffer(*inputImage2, inputImageRect2);
        filterContext.drawImageBuffer(*inputImage, inputImageRect);
        break;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_IN: {
        // Applies only to the intersected region.
        IntRect destinationRect = input.absoluteImageRect();
        destinationRect.intersect(input2.absoluteImageRect());
        destinationRect.intersect(result.absoluteImageRect());
        if (destinationRect.isEmpty())
            break;
        IntRect adjustedDestinationRect = destinationRect - result.absoluteImageRect().location();
        IntRect sourceRect = destinationRect - input.absoluteImageRect().location();
        IntRect source2Rect = destinationRect - input2.absoluteImageRect().location();
        filterContext.drawImageBuffer(*inputImage2, FloatRect(adjustedDestinationRect), FloatRect(source2Rect));
        filterContext.drawImageBuffer(*inputImage, FloatRect(adjustedDestinationRect), FloatRect(sourceRect), { CompositeOperator::SourceIn });
        break;
    }

    case CompositeOperationType::FECOMPOSITE_OPERATOR_OUT:
        filterContext.drawImageBuffer(*inputImage, inputImageRect);
        filterContext.drawImageBuffer(*inputImage2, inputImageRect2, { { }, inputImage2->logicalSize() }, { CompositeOperator::DestinationOut });
        break;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_ATOP:
        filterContext.drawImageBuffer(*inputImage2, inputImageRect2);
        filterContext.drawImageBuffer(*inputImage, inputImageRect, { { }, inputImage->logicalSize() }, { CompositeOperator::SourceAtop });
        break;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_XOR:
        filterContext.drawImageBuffer(*inputImage2, inputImageRect2);
        filterContext.drawImageBuffer(*inputImage, inputImageRect, { { }, inputImage->logicalSize() }, { CompositeOperator::XOR });
        break;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_ARITHMETIC:
        ASSERT_NOT_REACHED();
        return false;

    case CompositeOperationType::FECOMPOSITE_OPERATOR_LIGHTER:
        filterContext.drawImageBuffer(*inputImage2, inputImageRect2);
        filterContext.drawImageBuffer(*inputImage, inputImageRect, { { }, inputImage->logicalSize() }, { CompositeOperator::PlusLighter });
        break;
    }

    return true;
}

} // namespace WebCore
