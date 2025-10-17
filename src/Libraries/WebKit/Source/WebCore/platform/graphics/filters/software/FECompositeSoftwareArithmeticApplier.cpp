/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include "FECompositeSoftwareArithmeticApplier.h"

#if !HAVE(ARM_NEON_INTRINSICS)

#include "FEComposite.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PixelBuffer.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FECompositeSoftwareArithmeticApplier);

FECompositeSoftwareArithmeticApplier::FECompositeSoftwareArithmeticApplier(const FEComposite& effect)
    : Base(effect)
{
    ASSERT(m_effect.operation() == CompositeOperationType::FECOMPOSITE_OPERATOR_ARITHMETIC);
}

uint8_t FECompositeSoftwareArithmeticApplier::clampByte(int c)
{
    std::array<uint8_t, 3> buff { static_cast<uint8_t>(c), 255, 0 };
    unsigned uc = static_cast<unsigned>(c);
    return buff[!!(uc & ~0xff) + !!(uc & ~(~0u >> 1))];
}

template <int b1, int b4>
inline void FECompositeSoftwareArithmeticApplier::computePixels(std::span<unsigned char> source, std::span<unsigned char> destination, int pixelArrayLength, float k1, float k2, float k3, float k4)
{
    float scaledK1;
    float scaledK4;
    if (b1)
        scaledK1 = k1 / 255.0f;
    if (b4)
        scaledK4 = k4 * 255.0f;

    for (int index = 0; index < pixelArrayLength; ++index) {
        unsigned char i1 = source[index];
        unsigned char& i2 = destination[index];
        float result = k2 * i1 + k3 * i2;
        if (b1)
            result += scaledK1 * i1 * i2;
        if (b4)
            result += scaledK4;

        i2 = clampByte(result);
    }
}

// computePixelsUnclamped is a faster version of computePixels for the common case where clamping
// is not necessary. This enables aggresive compiler optimizations such as auto-vectorization.
template <int b1, int b4>
inline void FECompositeSoftwareArithmeticApplier::computePixelsUnclamped(std::span<unsigned char> source, std::span<unsigned char> destination, int pixelArrayLength, float k1, float k2, float k3, float k4)
{
    float scaledK1;
    float scaledK4;
    if (b1)
        scaledK1 = k1 / 255.0f;
    if (b4)
        scaledK4 = k4 * 255.0f;

    for (int index = 0; index < pixelArrayLength; ++index) {
        unsigned char i1 = source[index];
        unsigned char& i2 = destination[index];
        float result = k2 * i1 + k3 * i2;
        if (b1)
            result += scaledK1 * i1 * i2;
        if (b4)
            result += scaledK4;

        i2 = result;
    }
}

inline void FECompositeSoftwareArithmeticApplier::applyPlatform(std::span<unsigned char> source, std::span<unsigned char> destination, int pixelArrayLength, float k1, float k2, float k3, float k4)
{
    float upperLimit = std::max(0.0f, k1) + std::max(0.0f, k2) + std::max(0.0f, k3) + k4;
    float lowerLimit = std::min(0.0f, k1) + std::min(0.0f, k2) + std::min(0.0f, k3) + k4;
    if ((k4 >= 0.0f && k4 <= 1.0f) && (upperLimit >= 0.0f && upperLimit <= 1.0f) && (lowerLimit >= 0.0f && lowerLimit <= 1.0f)) {
        if (k4) {
            if (k1)
                computePixelsUnclamped<1, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
            else
                computePixelsUnclamped<0, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        } else {
            if (k1)
                computePixelsUnclamped<1, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
            else
                computePixelsUnclamped<0, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        }
        return;
    }

    if (k4) {
        if (k1)
            computePixels<1, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        else
            computePixels<0, 1>(source, destination, pixelArrayLength, k1, k2, k3, k4);
    } else {
        if (k1)
            computePixels<1, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
        else
            computePixels<0, 0>(source, destination, pixelArrayLength, k1, k2, k3, k4);
    }
}

bool FECompositeSoftwareArithmeticApplier::apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const
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

    auto sourcePixelBytes = sourcePixelBuffer->bytes();
    auto destinationPixelBytes = destinationPixelBuffer->bytes();

    auto length = sourcePixelBuffer->bytes().size();
    ASSERT(length == destinationPixelBuffer->bytes().size());

    applyPlatform(sourcePixelBytes, destinationPixelBytes, length, m_effect.k1(), m_effect.k2(), m_effect.k3(), m_effect.k4());
    return true;
}

} // namespace WebCore

#endif // !HAVE(ARM_NEON_INTRINSICS)
