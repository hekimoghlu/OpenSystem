/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include "ImageBufferCairoBackend.h"

#include "BitmapImage.h"
#include "CairoOperations.h"
#include "Color.h"
#include "ColorTransferFunctions.h"
#include "GraphicsContext.h"
#include "GraphicsContextCairo.h"
#include <cairo.h>

#if USE(CAIRO)

namespace WebCore {

void ImageBufferCairoBackend::transformToColorSpace(const DestinationColorSpace& newColorSpace)
{
    if (m_parameters.colorSpace == newColorSpace)
        return;

    // only sRGB <-> linearRGB are supported at the moment
    if ((m_parameters.colorSpace != DestinationColorSpace::LinearSRGB() && m_parameters.colorSpace != DestinationColorSpace::SRGB())
        || (newColorSpace != DestinationColorSpace::LinearSRGB() && newColorSpace != DestinationColorSpace::SRGB()))
        return;

    m_parameters.colorSpace = newColorSpace;

    if (newColorSpace == DestinationColorSpace::LinearSRGB()) {
        static const std::array<uint8_t, 256> linearRgbLUT = [] {
            std::array<uint8_t, 256> array;
            for (unsigned i = 0; i < 256; i++) {
                float color = i / 255.0f;
                color = SRGBTransferFunction<float, TransferFunctionMode::Clamped>::toLinear(color);
                array[i] = static_cast<uint8_t>(round(color * 255));
            }
            return array;
        }();
        platformTransformColorSpace(linearRgbLUT);
    } else {
        static const std::array<uint8_t, 256> deviceRgbLUT= [] {
            std::array<uint8_t, 256> array;
            for (unsigned i = 0; i < 256; i++) {
                float color = i / 255.0f;
                color = SRGBTransferFunction<float, TransferFunctionMode::Clamped>::toGammaEncoded(color);
                array[i] = static_cast<uint8_t>(round(color * 255));
            }
            return array;
        }();
        platformTransformColorSpace(deviceRgbLUT);
    }
}

} // namespace WebCore

#endif // USE(CAIRO)
