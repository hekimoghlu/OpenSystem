/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#include "FEMorphologySoftwareApplier.h"

#include "FEMorphology.h"
#include "Filter.h"
#include "PixelBuffer.h"
#include <wtf/ParallelJobs.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEMorphologySoftwareApplier);

inline ColorComponents<uint8_t, 4> FEMorphologySoftwareApplier::minOrMax(const ColorComponents<uint8_t, 4>& a, const ColorComponents<uint8_t, 4>& b, MorphologyOperatorType type)
{
    if (type == MorphologyOperatorType::Erode)
        return perComponentMin(a, b);

    return perComponentMax(a, b);
}

inline ColorComponents<uint8_t, 4> FEMorphologySoftwareApplier::columnExtremum(const PixelBuffer& srcPixelBuffer, int x, int yStart, int yEnd, int width, MorphologyOperatorType type)
{
    auto extremum = makeColorComponentsfromPixelValue(PackedColor::RGBA { reinterpretCastSpanStartTo<const unsigned>(srcPixelBuffer.bytes().subspan(pixelArrayIndex(x, yStart, width))) });

    for (int y = yStart + 1; y < yEnd; ++y) {
        auto pixel = makeColorComponentsfromPixelValue(PackedColor::RGBA { reinterpretCastSpanStartTo<const unsigned>(srcPixelBuffer.bytes().subspan(pixelArrayIndex(x, y, width))) });
        extremum = minOrMax(extremum, pixel, type);
    }
    return extremum;
}

inline ColorComponents<uint8_t, 4> FEMorphologySoftwareApplier::kernelExtremum(const ColumnExtrema& kernel, MorphologyOperatorType type)
{
    auto extremum = kernel[0];
    for (size_t i = 1; i < kernel.size(); ++i)
        extremum = minOrMax(extremum, kernel[i], type);

    return extremum;
}

void FEMorphologySoftwareApplier::applyPlatformGeneric(const PaintingData& paintingData, int startY, int endY)
{
    ASSERT(endY > startY);

    const auto& srcPixelBuffer = *paintingData.srcPixelBuffer;
    auto& dstPixelBuffer = *paintingData.dstPixelBuffer;

    const int radiusX = paintingData.radiusX;
    const int radiusY = paintingData.radiusY;
    const int width = paintingData.width;
    const int height = paintingData.height;

    ASSERT(radiusX <= width || radiusY <= height);
    ASSERT(startY >= 0 && endY <= height && startY < endY);

    ColumnExtrema extrema;
    extrema.reserveInitialCapacity(2 * radiusX + 1);

    for (int y = startY; y < endY; ++y) {
        int yRadiusStart = std::max(0, y - radiusY);
        int yRadiusEnd = std::min(height, y + radiusY + 1);

        extrema.shrink(0);

        // We start at the left edge, so compute extreme for the radiusX columns.
        for (int x = 0; x < radiusX; ++x)
            extrema.append(columnExtremum(srcPixelBuffer, x, yRadiusStart, yRadiusEnd, width, paintingData.type));

        // Kernel is filled, get extrema of next column
        for (int x = 0; x < width; ++x) {
            if (x < width - radiusX)
                extrema.append(columnExtremum(srcPixelBuffer, x + radiusX, yRadiusStart, yRadiusEnd, width, paintingData.type));

            if (x > radiusX)
                extrema.remove(0);

            unsigned& destPixel = reinterpretCastSpanStartTo<unsigned>(dstPixelBuffer.bytes().subspan(pixelArrayIndex(x, y, width)));
            destPixel = makePixelValueFromColorComponents(kernelExtremum(extrema, paintingData.type)).value;
        }
    }
}

void FEMorphologySoftwareApplier::applyPlatformWorker(ApplyParameters* params)
{
    applyPlatformGeneric(*params->paintingData, params->startY, params->endY);
}

void FEMorphologySoftwareApplier::applyPlatform(const PaintingData& paintingData)
{
    // Empirically, runtime is approximately linear over reasonable kernel sizes with a slope of about 0.65.
    float kernelFactor = sqrt(paintingData.radiusX * paintingData.radiusY) * 0.65;

    static const int minimalArea = (160 * 160); // Empirical data limit for parallel jobs
    
    unsigned maxNumThreads = paintingData.height / 8;
    unsigned optimalThreadNumber = std::min<unsigned>((paintingData.width * paintingData.height * kernelFactor) / minimalArea, maxNumThreads);
    if (optimalThreadNumber > 1) {
        ParallelJobs<ApplyParameters> parallelJobs(&applyPlatformWorker, optimalThreadNumber);
        auto numOfThreads = parallelJobs.numberOfJobs();
        if (numOfThreads > 1) {
            // Split the job into "jobSize"-sized jobs but there a few jobs that need to be slightly larger since
            // jobSize * jobs < total size. These extras are handled by the remainder "jobsWithExtra".
            int jobSize = paintingData.height / numOfThreads;
            int jobsWithExtra = paintingData.height % numOfThreads;
            int currentY = 0;
            for (int job = numOfThreads - 1; job >= 0; --job) {
                ApplyParameters& param = parallelJobs.parameter(job);
                param.startY = currentY;
                currentY += job < jobsWithExtra ? jobSize + 1 : jobSize;
                param.endY = currentY;
                param.paintingData = &paintingData;
            }
            parallelJobs.execute();
            return;
        }
        // Fallback to single thread model
    }

    applyPlatformGeneric(paintingData, 0, paintingData.height);
}

bool FEMorphologySoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    auto destinationPixelBuffer = result.pixelBuffer(AlphaPremultiplication::Premultiplied);
    if (!destinationPixelBuffer)
        return false;

    auto isDegenerate = [](const IntSize& absoluteRadius) -> bool {
        return absoluteRadius.width() < 0 || absoluteRadius.height() < 0 || absoluteRadius.isZero();
    };

    auto effectDrawingRect = result.absoluteImageRectRelativeTo(input);

    auto radius = filter.resolvedSize({ m_effect.radiusX(), m_effect.radiusY() });
    auto absoluteRadius = flooredIntSize(filter.scaledByFilterScale(radius));

    if (isDegenerate(absoluteRadius)) {
        input.copyPixelBuffer(*destinationPixelBuffer, effectDrawingRect);
        return true;
    }

    int radiusX = std::min(effectDrawingRect.width() - 1, absoluteRadius.width());
    int radiusY = std::min(effectDrawingRect.height() - 1, absoluteRadius.height());

    if (isDegenerate({ radiusX, radiusY })) {
        input.copyPixelBuffer(*destinationPixelBuffer, effectDrawingRect);
        return true;
    }

    auto sourcePixelBuffer = input.getPixelBuffer(AlphaPremultiplication::Premultiplied, effectDrawingRect, m_effect.operatingColorSpace());
    if (!sourcePixelBuffer)
        return false;

    PaintingData paintingData;
    paintingData.type = m_effect.morphologyOperator();
    paintingData.srcPixelBuffer = &*sourcePixelBuffer;
    paintingData.dstPixelBuffer = destinationPixelBuffer;
    paintingData.width = effectDrawingRect.width();
    paintingData.height = effectDrawingRect.height();
    paintingData.radiusX = radiusX;
    paintingData.radiusY = radiusY;

    applyPlatform(paintingData);
    return true;
}

} // namespace WebCore
