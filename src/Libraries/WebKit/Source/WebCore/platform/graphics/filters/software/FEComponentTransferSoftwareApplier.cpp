/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include "FEComponentTransferSoftwareApplier.h"

#include "FEComponentTransfer.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PixelBuffer.h"
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEComponentTransferSoftwareApplier);

void FEComponentTransferSoftwareApplier::applyPlatform(PixelBuffer& pixelBuffer) const
{
    auto data = pixelBuffer.bytes();
    auto pixelByteLength = pixelBuffer.bytes().size();

    auto redTable   = FEComponentTransfer::computeLookupTable(m_effect.redFunction());
    auto greenTable = FEComponentTransfer::computeLookupTable(m_effect.greenFunction());
    auto blueTable  = FEComponentTransfer::computeLookupTable(m_effect.blueFunction());
    auto alphaTable = FEComponentTransfer::computeLookupTable(m_effect.alphaFunction());

    for (unsigned pixelOffset = 0; pixelOffset < pixelByteLength; pixelOffset += 4) {
        data[pixelOffset]     = redTable[data[pixelOffset]];
        data[pixelOffset + 1] = greenTable[data[pixelOffset + 1]];
        data[pixelOffset + 2] = blueTable[data[pixelOffset + 2]];
        data[pixelOffset + 3] = alphaTable[data[pixelOffset + 3]];
    }
}

bool FEComponentTransferSoftwareApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();
    
    auto destinationPixelBuffer = result.pixelBuffer(AlphaPremultiplication::Unpremultiplied);
    if (!destinationPixelBuffer)
        return false;

    auto drawingRect = result.absoluteImageRectRelativeTo(input);
    input.copyPixelBuffer(*destinationPixelBuffer, drawingRect);

    applyPlatform(*destinationPixelBuffer);
    return true;
}

} // namespace WebCore
