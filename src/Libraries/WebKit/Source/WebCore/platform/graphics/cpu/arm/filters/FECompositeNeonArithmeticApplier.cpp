/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#include "FECompositeNeonArithmeticApplier.h"

#if HAVE(ARM_NEON_INTRINSICS)

#include "FEComposite.h"
#include "NEONHelpers.h"
#include <arm_neon.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FECompositeNeonArithmeticApplier);

FECompositeNeonArithmeticApplier::FECompositeNeonArithmeticApplier(const FEComposite& effect)
    : Base(effect)
{
    ASSERT(m_effect.operation() == CompositeOperationType::FECOMPOSITE_OPERATOR_ARITHMETIC);
}

template <int b1, int b4>
inline void FECompositeNeonArithmeticApplier::computePixels(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4)
{
    float32x4_t k1x4 = vdupq_n_f32(k1 / 255);
    float32x4_t k2x4 = vdupq_n_f32(k2);
    float32x4_t k3x4 = vdupq_n_f32(k3);
    float32x4_t k4x4 = vdupq_n_f32(k4 * 255);
    uint32x4_t max255 = vdupq_n_u32(255);

    const uint32_t* sourcePixel = reinterpret_cast<const uint32_t*>(source);
    uint32_t* destinationPixel = reinterpret_cast<uint32_t*>(destination);
    uint32_t* destinationEndPixel = destinationPixel + (pixelArrayLength >> 2);

    while (destinationPixel < destinationEndPixel) {
        float32x4_t sourcePixelAsFloat = loadRGBA8AsFloat(sourcePixel);
        float32x4_t destinationPixelAsFloat = loadRGBA8AsFloat(destinationPixel);

        float32x4_t result = vmulq_f32(sourcePixelAsFloat, k2x4);
        result = vmlaq_f32(result, destinationPixelAsFloat, k3x4);
        if (b1)
            result = vmlaq_f32(result, vmulq_f32(sourcePixelAsFloat, destinationPixelAsFloat), k1x4);
        if (b4)
            result = vaddq_f32(result, k4x4);

        // Convert result to uint so negative values are converted to zero.
        uint16x4_t temporary3 = vmovn_u32(vminq_u32(vcvtq_u32_f32(result), max255));
        uint8x8_t temporary4 = vmovn_u16(vcombine_u16(temporary3, temporary3));
        *destinationPixel++ = vget_lane_u32(vreinterpret_u32_u8(temporary4), 0);
        ++sourcePixel;
    }
}

inline void FECompositeNeonArithmeticApplier::applyPlatform(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4)
{
    if (!k4) {
        if (!k1) {
            computePixels<0, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
            return;
        }

        computePixels<1, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        return;
    }

    if (!k1) {
        computePixels<0, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        return;
    }
    computePixels<1, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
}

bool FECompositeNeonArithmeticApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();
    auto& input2 = inputs[1].get();

    auto destinationPixelBuffer = result.pixelBuffer(AlphaPremultiplication::Premultiplied);
    if (!destinationPixelBuffer)
        return false;

    IntRect effectADrawingRect = result.absoluteImageRectRelativeTo(input);
    auto sourcePixelBuffer = input.getPixelBuffer(AlphaPremultiplication::Premultiplied, effectADrawingRect, m_effect.operatingColorSpace());
    if (!sourcePixelBuffer)
        return false;

    IntRect effectBDrawingRect = result.absoluteImageRectRelativeTo(input2);
    input2.copyPixelBuffer(*destinationPixelBuffer, effectBDrawingRect);

    auto* sourcePixelBytes = sourcePixelBuffer->bytes().data();
    auto* destinationPixelBytes = destinationPixelBuffer->bytes().data();

    auto length = sourcePixelBuffer->bytes().size();
    ASSERT(length == destinationPixelBuffer->bytes().size());

    applyPlatform(sourcePixelBytes, destinationPixelBytes, length, m_effect.k1(), m_effect.k2(), m_effect.k3(), m_effect.k4());
    return true;
}

} // namespace WebCore

#endif // HAVE(ARM_NEON_INTRINSICS)
