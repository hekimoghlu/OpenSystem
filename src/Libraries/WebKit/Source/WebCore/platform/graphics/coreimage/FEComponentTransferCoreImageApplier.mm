/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#import "FEComponentTransferCoreImageApplier.h"

#if USE(CORE_IMAGE)

#import "FEComponentTransfer.h"
#import <CoreImage/CIContext.h>
#import <CoreImage/CIFilter.h>
#import <CoreImage/CoreImage.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEComponentTransferCoreImageApplier);

FEComponentTransferCoreImageApplier::FEComponentTransferCoreImageApplier(const FEComponentTransfer& effect)
    : Base(effect)
{
    // FIXME: Implement the rest of FEComponentTransfer functions
    ASSERT(supportsCoreImageRendering(effect));
}

bool FEComponentTransferCoreImageApplier::supportsCoreImageRendering(const FEComponentTransfer& effect)
{
    auto isNullOrLinear = [] (const ComponentTransferFunction& function) {
        return function.type == ComponentTransferType::FECOMPONENTTRANSFER_TYPE_UNKNOWN
            || function.type == ComponentTransferType::FECOMPONENTTRANSFER_TYPE_LINEAR;
    };

    return isNullOrLinear(effect.redFunction())
        && isNullOrLinear(effect.greenFunction())
        && isNullOrLinear(effect.blueFunction())
        && isNullOrLinear(effect.alphaFunction());
}

bool FEComponentTransferCoreImageApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    ASSERT(inputs.size() == 1);
    auto& input = inputs[0].get();

    auto inputImage = input.ciImage();
    if (!inputImage)
        return false;

    auto componentTransferFilter = [CIFilter filterWithName:@"CIColorPolynomial"];
    [componentTransferFilter setValue:inputImage.get() forKey:kCIInputImageKey];

    auto setCoefficients = [&] (NSString *key, const ComponentTransferFunction& function) {
        if (function.type == ComponentTransferType::FECOMPONENTTRANSFER_TYPE_LINEAR)
            [componentTransferFilter setValue:[CIVector vectorWithX:function.intercept Y:function.slope Z:0 W:0] forKey:key];
    };

    setCoefficients(@"inputRedCoefficients", m_effect.redFunction());
    setCoefficients(@"inputGreenCoefficients", m_effect.greenFunction());
    setCoefficients(@"inputBlueCoefficients", m_effect.blueFunction());
    setCoefficients(@"inputAlphaCoefficients", m_effect.alphaFunction());

    result.setCIImage(componentTransferFilter.outputImage);
    return true;
}

} // namespace WebCore

#endif // USE(CORE_IMAGE)
