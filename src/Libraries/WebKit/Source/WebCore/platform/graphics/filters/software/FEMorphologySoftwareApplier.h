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
#pragma once

#include "ColorComponents.h"
#include "ColorTypes.h"
#include "FilterEffectApplier.h"
#include "PixelBuffer.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEMorphology;
enum class MorphologyOperatorType : uint8_t;

class FEMorphologySoftwareApplier final : public FilterEffectConcreteApplier<FEMorphology> {
    WTF_MAKE_TZONE_ALLOCATED(FEMorphologySoftwareApplier);
    using Base = FilterEffectConcreteApplier<FEMorphology>;

public:
    using Base::Base;

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    using ColumnExtrema = Vector<ColorComponents<uint8_t, 4>, 16>;

    struct PaintingData {
        MorphologyOperatorType type;
        int radiusX;
        int radiusY;
        const PixelBuffer* srcPixelBuffer;
        PixelBuffer* dstPixelBuffer;
        int width;
        int height;
    };

    struct ApplyParameters {
        const PaintingData* paintingData;
        int startY;
        int endY;
    };

    static inline int pixelArrayIndex(int x, int y, int width) { return (y * width + x) * 4; }
    static inline PackedColor::RGBA makePixelValueFromColorComponents(const ColorComponents<uint8_t, 4>& components) { return PackedColor::RGBA { makeFromComponents<SRGBA<uint8_t>>(components) }; }

    static inline ColorComponents<uint8_t, 4> makeColorComponentsfromPixelValue(PackedColor::RGBA pixel) { return asColorComponents(asSRGBA(pixel).resolved()); }
    static inline ColorComponents<uint8_t, 4> minOrMax(const ColorComponents<uint8_t, 4>& a, const ColorComponents<uint8_t, 4>& b, MorphologyOperatorType);
    static inline ColorComponents<uint8_t, 4> columnExtremum(const PixelBuffer& srcPixelBuffer, int x, int yStart, int yEnd, int width, MorphologyOperatorType);
    static inline ColorComponents<uint8_t, 4> kernelExtremum(const ColumnExtrema& kernel, MorphologyOperatorType);

    static void applyPlatformGeneric(const PaintingData&, int startY, int endY);
    static void applyPlatformWorker(ApplyParameters*);
    static void applyPlatform(const PaintingData&);
};

} // namespace WebCore
