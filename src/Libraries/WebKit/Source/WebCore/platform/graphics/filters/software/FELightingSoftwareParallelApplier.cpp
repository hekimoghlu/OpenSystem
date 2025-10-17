/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#include "FELightingSoftwareParallelApplier.h"

#if !(CPU(ARM_NEON) && CPU(ARM_TRADITIONAL) && COMPILER(GCC_COMPATIBLE))
#include "FELightingSoftwareApplierInlines.h"
#include "ImageBuffer.h"
#include <wtf/ParallelJobs.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FELightingSoftwareParallelApplier);

// This appears to read from and write to the same pixel buffer, but it only reads the alpha channel, and writes the non-alpha channels.
void FELightingSoftwareParallelApplier::applyPlatformPaint(const LightingData& data, const LightSource::PaintingData& paintingData, int startY, int endY)
{
    // Make sure startY is > 0 since we read from the previous row in the loop.
    ASSERT(startY);
    ASSERT(endY > startY);

    for (int y = startY; y < endY; ++y) {
        int rowStartOffset = y * data.widthMultipliedByPixelSize;
        int previousRowStart = rowStartOffset - data.widthMultipliedByPixelSize;
        int nextRowStart = rowStartOffset + data.widthMultipliedByPixelSize;

        // alphaWindow is a local cache of alpha values.
        // Fill the two right columns putting the left edge value in the center column.
        // For each pixel, we shift each row left then fill the right column.
        AlphaWindow alphaWindow;
        alphaWindow.setTop(data.pixels->item(previousRowStart + cAlphaChannelOffset));
        alphaWindow.setTopRight(data.pixels->item(previousRowStart + cPixelSize + cAlphaChannelOffset));

        alphaWindow.setCenter(data.pixels->item(rowStartOffset + cAlphaChannelOffset));
        alphaWindow.setRight(data.pixels->item(rowStartOffset + cPixelSize + cAlphaChannelOffset));

        alphaWindow.setBottom(data.pixels->item(nextRowStart + cAlphaChannelOffset));
        alphaWindow.setBottomRight(data.pixels->item(nextRowStart + cPixelSize + cAlphaChannelOffset));

        int offset = rowStartOffset + cPixelSize;
        for (int x = 1; x < data.width - 1; ++x, offset += cPixelSize) {
            alphaWindow.shift();
            setPixelInternal(offset, data, paintingData, x, y, cFactor1div4, cFactor1div4, data.interiorNormal(offset, alphaWindow), alphaWindow.center());
        }
    }
}

void FELightingSoftwareParallelApplier::applyPlatformWorker(ApplyParameters* parameters)
{
    applyPlatformPaint(parameters->data, parameters->paintingData, parameters->yStart, parameters->yEnd);
}

void FELightingSoftwareParallelApplier::applyPlatformParallel(const LightingData& data, const LightSource::PaintingData& paintingData) const
{
    unsigned rowsToProcess = data.height - 2;
    unsigned maxNumThreads = rowsToProcess / 8;
    unsigned optimalThreadNumber = std::min<unsigned>(((data.width - 2) * rowsToProcess) / minimalRectDimension, maxNumThreads);

    if (optimalThreadNumber > 1) {
        // Initialize parallel jobs
        ParallelJobs<ApplyParameters> parallelJobs(&applyPlatformWorker, optimalThreadNumber);

        // Fill the parameter array
        int job = parallelJobs.numberOfJobs();
        if (job > 1) {
            // Split the job into "yStep"-sized jobs but there a few jobs that need to be slightly larger since
            // yStep * jobs < total size. These extras are handled by the remainder "jobsWithExtra".
            const int yStep = rowsToProcess / job;
            const int jobsWithExtra = rowsToProcess % job;

            int yStart = 1;
            for (--job; job >= 0; --job) {
                ApplyParameters& params = parallelJobs.parameter(job);
                params.data = data;
                params.paintingData = paintingData;
                params.yStart = yStart;
                yStart += job < jobsWithExtra ? yStep + 1 : yStep;
                params.yEnd = yStart;
            }
            parallelJobs.execute();
            return;
        }
        // Fallback to single threaded mode.
    }

    applyPlatformPaint(data, paintingData, 1, data.height - 1);
}

} // namespace WebCore

#endif // !(CPU(ARM_NEON) && CPU(ARM_TRADITIONAL) && COMPILER(GCC_COMPATIBLE))
