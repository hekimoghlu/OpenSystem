/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#include "Color.h"
#include "FilterEffect.h"
#include "FilterEffectApplier.h"
#include "FilterImageVector.h"
#include "LightSource.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FELighting;

class FELightingSoftwareApplier : public FilterEffectConcreteApplier<FELighting> {
    WTF_MAKE_TZONE_ALLOCATED(FELightingSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FELighting>;

protected:
    using Base::Base;

    static constexpr int minimalRectDimension = 100 * 100; // Empirical data limit for parallel jobs
    static constexpr int cPixelSize = 4;
    static constexpr int cAlphaChannelOffset = 3;
    static constexpr uint8_t cOpaqueAlpha = static_cast<uint8_t>(0xFF);

    // These factors and the normal coefficients come from the table under https://www.w3.org/TR/SVG/filters.html#feDiffuseLightingElement.
    static constexpr float cFactor1div2 = -1 / 2.f;
    static constexpr float cFactor1div3 = -1 / 3.f;
    static constexpr float cFactor1div4 = -1 / 4.f;
    static constexpr float cFactor2div3 = -2 / 3.f;

    struct AlphaWindow {
        std::array<std::array<uint8_t, 3>, 3> alpha = { };
    
        // The implementations are lined up to make comparing indices easier.
        uint8_t topLeft() const             { return alpha[0][0]; }
        uint8_t left() const                { return alpha[1][0]; }
        uint8_t bottomLeft() const          { return alpha[2][0]; }

        uint8_t top() const                 { return alpha[0][1]; }
        uint8_t center() const              { return alpha[1][1]; }
        uint8_t bottom() const              { return alpha[2][1]; }

        void setTop(uint8_t value)          { alpha[0][1] = value; }
        void setCenter(uint8_t value)       { alpha[1][1] = value; }
        void setBottom(uint8_t value)       { alpha[2][1] = value; }

        void setTopRight(uint8_t value)     { alpha[0][2] = value; }
        void setRight(uint8_t value)        { alpha[1][2] = value; }
        void setBottomRight(uint8_t value)  { alpha[2][2] = value; }

        static void shiftRow(std::array<uint8_t, 3>& alpha)
        {
            alpha[0] = alpha[1];
            alpha[1] = alpha[2];
        }
    
        void shift()
        {
            shiftRow(alpha[0]);
            shiftRow(alpha[1]);
            shiftRow(alpha[2]);
        }
    };

    struct LightingData {
        // This structure contains only read-only (SMP safe) data
        const Filter* filter;
        const FilterImage* result;
        FilterEffect::Type filterType;
        Color lightingColor;
        float surfaceScale;
        float diffuseConstant;
        float specularConstant;
        float specularExponent;
        const LightSource* lightSource;
        const DestinationColorSpace* operatingColorSpace;

        PixelBuffer* pixels;
        int widthMultipliedByPixelSize;
        int width;
        int height;

        inline IntSize topLeftNormal(int offset) const;
        inline IntSize topRowNormal(int offset) const;
        inline IntSize topRightNormal(int offset) const;
        inline IntSize leftColumnNormal(int offset) const;
        inline IntSize interiorNormal(int offset, AlphaWindow&) const;
        inline IntSize rightColumnNormal(int offset) const;
        inline IntSize bottomLeftNormal(int offset) const;
        inline IntSize bottomRowNormal(int offset) const;
        inline IntSize bottomRightNormal(int offset) const;
    };

    static void setPixelInternal(int offset, const LightingData&, const LightSource::PaintingData&, int x, int y, float factorX, float factorY, IntSize normal2DVector, float alpha);
    static void setPixel(int offset, const LightingData&, const LightSource::PaintingData&, int x, int y, float factorX, float factorY, IntSize normal2DVector);

    virtual void applyPlatformParallel(const LightingData&, const LightSource::PaintingData&) const = 0;
    void applyPlatform(const LightingData&) const;
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;
};

} // namespace WebCore
