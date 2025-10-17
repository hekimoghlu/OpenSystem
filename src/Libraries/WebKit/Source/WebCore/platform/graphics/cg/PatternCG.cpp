/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "Pattern.h"

#if USE(CG)

#include "AffineTransform.h"
#include "CGUtilities.h"
#include "GraphicsContextCG.h"
#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <wtf/MainThread.h>

namespace WebCore {

static void patternCallback(void* info, CGContextRef context)
{
    CGImageRef platformImage = static_cast<CGImageRef>(info);
    if (!platformImage)
        return;
    auto rect = cgRoundToDevicePixels(CGContextGetUserSpaceToDeviceSpaceTransform(context), cgImageRect(platformImage));
    CGContextDrawImage(context, rect, platformImage);
}

static void patternReleaseCallback(void* info)
{
    callOnMainThread([image = adoptCF(static_cast<CGImageRef>(info))] { });
}

RetainPtr<CGPatternRef> Pattern::createPlatformPattern(const AffineTransform& userSpaceTransform) const
{
    auto nativeImage = tileNativeImage();
    if (!nativeImage)
        return nullptr;

    auto platformImage = nativeImage->platformImage();
    if (!platformImage)
        return nullptr;

    FloatRect tileRect = { { }, nativeImage->size() };

    auto patternTransform = userSpaceTransform * patternSpaceTransform();
    patternTransform.scaleNonUniform(1, -1);
    patternTransform.translate(0, -tileRect.height());

    // If we're repeating in both directions, we can use image-backed patterns
    // instead of custom patterns, and avoid tiling-edge pixel cracks.
    if (repeatX() && repeatY())
        return adoptCF(CGPatternCreateWithImage2(platformImage.get(), patternTransform, kCGPatternTilingConstantSpacing));

    // If FLT_MAX should also be used for xStep or yStep, nothing is rendered. Using fractions of FLT_MAX also
    // result in nothing being rendered.
    // INT_MAX is almost correct, but there seems to be some number wrapping occurring making the fill
    // pattern is not filled correctly.
    // To make error of floating point less than 0.5, we use the half of the number of mantissa of float (1 << 22).
    CGFloat xStep = repeatX() ? tileRect.width() : (1 << 22);
    CGFloat yStep = repeatY() ? tileRect.height() : (1 << 22);

    // The pattern will release the CGImageRef when it's done rendering in patternReleaseCallback
    CGImageRef image = platformImage.leakRef();

    const CGPatternCallbacks patternCallbacks = { 0, patternCallback, patternReleaseCallback };
    return adoptCF(CGPatternCreate(image, tileRect, patternTransform, xStep, yStep, kCGPatternTilingConstantSpacing, TRUE, &patternCallbacks));
}

}

#endif
