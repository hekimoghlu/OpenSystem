/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#ifndef FEGaussianBlurNEON_h
#define FEGaussianBlurNEON_h

#if HAVE(ARM_NEON_INTRINSICS)

#include "FEGaussianBlur.h"
#include "NEONHelpers.h"

namespace WebCore {

inline void boxBlurNEON(const PixelBuffer& srcPixelBuffer, PixelBuffer& dstPixelBuffer,
                        unsigned dx, int dxLeft, int dxRight, int stride, int strideLine, int effectWidth, int effectHeight)
{
    const uint32_t* sourcePixel = reinterpret_cast<uint32_t*>(srcPixelBuffer.bytes().data());
    uint32_t* destinationPixel = reinterpret_cast<uint32_t*>(dstPixelBuffer.bytes().data());

    float32x4_t deltaX = vdupq_n_f32(1.0 / dx);
    int pixelLine = strideLine / 4;
    int pixelStride = stride / 4;

    for (int y = 0; y < effectHeight; ++y) {
        int line = y * pixelLine;
        float32x4_t sum = vdupq_n_f32(0);
        // Fill the kernel
        int maxKernelSize = std::min(dxRight, effectWidth);
        for (int i = 0; i < maxKernelSize; ++i) {
            float32x4_t sourcePixelAsFloat = loadRGBA8AsFloat(sourcePixel + line + i * pixelStride);
            sum = vaddq_f32(sum, sourcePixelAsFloat);
        }

        // Blurring
        for (int x = 0; x < effectWidth; ++x) {
            int pixelOffset = line + x * pixelStride;
            float32x4_t result = vmulq_f32(sum, deltaX);
            storeFloatAsRGBA8(result, destinationPixel + pixelOffset);
            if (x >= dxLeft) {
                float32x4_t sourcePixelAsFloat = loadRGBA8AsFloat(sourcePixel + pixelOffset - dxLeft * pixelStride);
                sum = vsubq_f32(sum, sourcePixelAsFloat);
            }
            if (x + dxRight < effectWidth) {
                float32x4_t sourcePixelAsFloat = loadRGBA8AsFloat(sourcePixel + pixelOffset + dxRight * pixelStride);
                sum = vaddq_f32(sum, sourcePixelAsFloat);
            }
        }
    }
}

} // namespace WebCore

#endif // HAVE(ARM_NEON_INTRINSICS)

#endif // FEGaussianBlurNEON_h
