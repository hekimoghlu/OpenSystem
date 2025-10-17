/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include "FEColorMatrixSkiaApplier.h"

#if USE(SKIA)

#include "FEColorMatrix.h"
#include "FilterImage.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "NativeImage.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib/Win port
#include <skia/core/SkCanvas.h>
#include <skia/core/SkColorFilter.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEColorMatrixSkiaApplier);

bool FEColorMatrixSkiaApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
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

    auto values = FEColorMatrix::normalizedFloats(m_effect.values());
    Vector<float> matrix;

    std::array<float, 9> components;

    switch (m_effect.type()) {
    case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX:
        matrix = values;
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
        FEColorMatrix::calculateSaturateComponents(components, values[0]);
        matrix = Vector<float>({
            components[0], components[1], components[2], 0.0, 0.0,
            components[3], components[4], components[5], 0.0, 0.0,
            components[6], components[7], components[8], 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
        });
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
        FEColorMatrix::calculateHueRotateComponents(components, values[0]);
        matrix = Vector<float>({
            components[0], components[1], components[2], 0.0, 0.0,
            components[3], components[4], components[5], 0.0, 0.0,
            components[6], components[7], components[8], 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
        });
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA:
        matrix = Vector<float>({
            0.0,    0.0,    0.0,    0.0, 0.0,
            0.0,    0.0,    0.0,    0.0, 0.0,
            0.0,    0.0,    0.0,    0.0, 0.0,
            0.2125, 0.7154, 0.0721, 0.0, 0.0,
        });
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN:
        return false;
    }

    SkPaint paint;
    paint.setColorFilter(SkColorFilters::Matrix(matrix.data()));

    auto inputOffsetWithinResult = input.absoluteImageRectRelativeTo(result).location();
    resultImage->context().platformContext()->drawImage(nativeImage->platformImage(), inputOffsetWithinResult.x(), inputOffsetWithinResult.y(), { }, &paint);
    return true;
}

} // namespace WebCore

#endif // USE(SKIA)
