/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#pragma once

#include "FilterEffectApplier.h"
#include "IntPoint.h"
#include "IntSize.h"
#include "PixelBuffer.h"
#include <JavaScriptCore/TypedArrayAdaptersForwardDeclarations.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class FEConvolveMatrix;
enum class EdgeModeType : uint8_t;

class FEConvolveMatrixSoftwareApplier final : public FilterEffectConcreteApplier<FEConvolveMatrix> {
    WTF_MAKE_TZONE_ALLOCATED(FEConvolveMatrixSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FEConvolveMatrix>;

public:
    FEConvolveMatrixSoftwareApplier(const FEConvolveMatrix& effect);

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    struct PaintingData {
        const PixelBuffer& sourcePixelBuffer;
        PixelBuffer& destinationPixelBuffer;
        int width;
        int height;

        IntSize kernelSize;
        float divisor;
        float bias;
        IntPoint targetOffset;
        EdgeModeType edgeMode;
        bool preserveAlpha;
        Vector<float> kernelMatrix;
    };

    static inline uint8_t clampRGBAValue(float channel, uint8_t max = 255);
    static inline void setDestinationPixels(const PixelBuffer& sourcePixelBuffer, PixelBuffer& destinationPixelBuffer, int& pixel, std::span<float> totals, float divisor, float bias, bool preserveAlphaValues);

    static inline int getPixelValue(const PaintingData&, int x, int y);

    static inline void setInteriorPixels(PaintingData&, int clipRight, int clipBottom, int yStart, int yEnd);
    static inline void setOuterPixels(PaintingData&, int x1, int y1, int x2, int y2);
    static void setInteriorPixels(PaintingData&, int clipRight, int clipBottom);
    void applyPlatform(PaintingData&) const;
};

} // namespace WebCore
