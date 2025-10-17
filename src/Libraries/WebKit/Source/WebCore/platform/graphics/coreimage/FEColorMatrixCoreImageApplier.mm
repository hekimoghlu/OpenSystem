/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#import "config.h"
#import "FEColorMatrixCoreImageApplier.h"

#if USE(CORE_IMAGE)

#import "FEColorMatrix.h"
#import "FilterImage.h"
#import <CoreImage/CIContext.h>
#import <CoreImage/CIFilter.h>
#import <CoreImage/CoreImage.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEColorMatrixCoreImageApplier);

FEColorMatrixCoreImageApplier::FEColorMatrixCoreImageApplier(const FEColorMatrix& effect)
    : Base(effect)
{
    // FIXME: Implement the rest of FEColorMatrix types
    ASSERT(supportsCoreImageRendering(effect));
}

bool FEColorMatrixCoreImageApplier::supportsCoreImageRendering(const FEColorMatrix& effect)
{
    return effect.type() == ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE
        || effect.type() == ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE
        || effect.type() == ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX;
}

bool FEColorMatrixCoreImageApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    ASSERT(inputs.size() == 1);
    auto& input = inputs[0].get();

    auto inputImage = input.ciImage();
    if (!inputImage)
        return false;

    auto values = FEColorMatrix::normalizedFloats(m_effect.values());
    std::array<float, 9> components;

    switch (m_effect.type()) {
    case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
        FEColorMatrix::calculateSaturateComponents(components, values[0]);
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
        FEColorMatrix::calculateHueRotateComponents(components, values[0]);
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX:
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN:
    case ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA: // FIXME: Add Luminance to Alpha Implementation
        return false;
    }

    auto *colorMatrixFilter = [CIFilter filterWithName:@"CIColorMatrix"];
    [colorMatrixFilter setValue:inputImage.get() forKey:kCIInputImageKey];

    switch (m_effect.type()) {
    case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
    case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
        [colorMatrixFilter setValue:[CIVector vectorWithX:components[0] Y:components[1] Z:components[2] W:0] forKey:@"inputRVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:components[3] Y:components[4] Z:components[5] W:0] forKey:@"inputGVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:components[6] Y:components[7] Z:components[8] W:0] forKey:@"inputBVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:0             Y:0             Z:0             W:1] forKey:@"inputAVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:0             Y:0             Z:0             W:0] forKey:@"inputBiasVector"];
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX:
        [colorMatrixFilter setValue:[CIVector vectorWithX:values[0]  Y:values[1]  Z:values[2]  W:values[3]]  forKey:@"inputRVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:values[5]  Y:values[6]  Z:values[7]  W:values[8]]  forKey:@"inputGVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:values[10] Y:values[11] Z:values[12] W:values[13]] forKey:@"inputBVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:values[15] Y:values[16] Z:values[17] W:values[18]] forKey:@"inputAVector"];
        [colorMatrixFilter setValue:[CIVector vectorWithX:values[4]  Y:values[9]  Z:values[14] W:values[19]] forKey:@"inputBiasVector"];
        break;

    case ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA:
    case ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN:
        return false;
    }

    result.setCIImage(colorMatrixFilter.outputImage);
    return true;
}

} // namespace WebCore

#endif // USE(CORE_IMAGE)
