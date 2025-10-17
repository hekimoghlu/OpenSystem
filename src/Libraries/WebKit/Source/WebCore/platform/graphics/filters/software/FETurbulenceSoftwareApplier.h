/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#include "FilterEffectApplier.h"
#include "FloatPoint.h"
#include "IntRect.h"
#include "PixelBuffer.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FETurbulence;
enum class TurbulenceType : uint8_t;

class FETurbulenceSoftwareApplier final : public FilterEffectConcreteApplier<FETurbulence> {
    WTF_MAKE_TZONE_ALLOCATED(FETurbulenceSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FETurbulence>;

public:
    using Base::Base;

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    // Produces results in the range [1, 2**31 - 2]. Algorithm is:
    // r = (a * r) mod m where a = s_randAmplitude = 16807 and
    // m = s_randMaximum = 2**31 - 1 = 2147483647, r = seed.
    // See [Park & Miller], CACM vol. 31 no. 10 p. 1195, Oct. 1988
    // To test: the algorithm should produce the result 1043618065
    // as the 10,000th generated number if the original seed is 1.
    static const int s_perlinNoise = 4096;
    static const long s_randMaximum = 2147483647; // 2**31 - 1
    static const int s_randAmplitude = 16807; // 7**5; primitive root of m
    static const int s_randQ = 127773; // m / a
    static const int s_randR = 2836; // m % a

    static const int s_blockSize = 256;
    static const int s_blockMask = s_blockSize - 1;

    struct PaintingData {
        // Compute pseudo random number.
        long random()
        {
            long result = s_randAmplitude * (seed % s_randQ) - s_randR * (seed / s_randQ);
            if (result <= 0)
                result += s_randMaximum;
            seed = result;
            return result;
        }

        TurbulenceType type;
        float baseFrequencyX;
        float baseFrequencyY;
        int numOctaves;
        long seed;
        bool stitchTiles;
        IntSize paintingSize;

        std::array<int, 2 * s_blockSize + 2> latticeSelector;
        std::array<std::array<std::array<float, 2>, 2 * s_blockSize + 2>, 4> gradient;
    };

    struct StitchData {
        int width { 0 }; // How much to subtract to wrap for stitching.
        int wrapX { 0 }; // Minimum value to wrap.
        int height { 0 };
        int wrapY { 0 };
    };

    struct ApplyParameters {
        IntRect filterRegion;
        FloatSize filterScale;
        PixelBuffer* pixelBuffer;
        PaintingData* paintingData;
        StitchData stitchData;
        int startY;
        int endY;
    };

    static inline float smoothCurve(float t) { return t * t * (3 - 2 * t); }
    static inline float linearInterpolation(float t, float a, float b) { return a + t * (b - a); }

    static PaintingData initPaintingData(TurbulenceType, float baseFrequencyX, float baseFrequencyY, int numOctaves, long seed, bool stitchTiles, const IntSize& paintingSize);
    static StitchData computeStitching(IntSize tileSize, float& baseFrequencyX, float& baseFrequencyY, bool stitchTiles);

    static ColorComponents<float, 4> noise2D(const PaintingData&, const StitchData&, const FloatPoint& noiseVector);
    static ColorComponents<uint8_t, 4> toIntBasedColorComponents(const ColorComponents<float, 4>& floatComponents);
    static ColorComponents<uint8_t, 4> calculateTurbulenceValueForPoint(const PaintingData&, StitchData, const FloatPoint&);

    static void applyPlatformGeneric(const IntRect& filterRegion, const FloatSize& filterScale, PixelBuffer&, const PaintingData&, StitchData, int startY, int endY);
    static void applyPlatformWorker(ApplyParameters*);
    static void applyPlatform(const IntRect& filterRegion, const FloatSize& filterScale, PixelBuffer&, PaintingData&, StitchData&);
};

} // namespace WebCore
