/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

#include "FloatRect.h"
#include "GraphicsTypes.h"
#include "IntRect.h"
#include <CoreGraphics/CoreGraphics.h>
#include <math.h>

namespace WebCore {

inline CGInterpolationQuality toCGInterpolationQuality(InterpolationQuality quality)
{
    switch (quality) {
    case InterpolationQuality::Default:
        return kCGInterpolationDefault;
    case InterpolationQuality::DoNotInterpolate:
        return kCGInterpolationNone;
    case InterpolationQuality::Low:
        return kCGInterpolationLow;
    case InterpolationQuality::Medium:
        return kCGInterpolationMedium;
    case InterpolationQuality::High:
        return kCGInterpolationHigh;
    }
    ASSERT_NOT_REACHED();
    return kCGInterpolationDefault;
}

inline FloatRect cgRoundToDevicePixelsNonIdentity(CGAffineTransform deviceMatrix, FloatRect rect)
{
    // It is not enough just to round to pixels in device space. The rotation part of the
    // affine transform matrix to device space can mess with this conversion if we have a
    // rotating image like the hands of the world clock widget. We just need the scale, so
    // we get the affine transform matrix and extract the scale.
    auto deviceScaleX = hypot(deviceMatrix.a, deviceMatrix.b);
    auto deviceScaleY = hypot(deviceMatrix.c, deviceMatrix.d);

    CGPoint deviceOrigin = CGPointMake(rect.x() * deviceScaleX, rect.y() * deviceScaleY);
    CGPoint deviceLowerRight = CGPointMake((rect.x() + rect.width()) * deviceScaleX,
        (rect.y() + rect.height()) * deviceScaleY);

    deviceOrigin.x = roundf(deviceOrigin.x);
    deviceOrigin.y = roundf(deviceOrigin.y);
    deviceLowerRight.x = roundf(deviceLowerRight.x);
    deviceLowerRight.y = roundf(deviceLowerRight.y);

    // Don't let the height or width round to 0 unless either was originally 0
    if (deviceOrigin.y == deviceLowerRight.y && rect.height())
        deviceLowerRight.y += 1;
    if (deviceOrigin.x == deviceLowerRight.x && rect.width())
        deviceLowerRight.x += 1;

    FloatPoint roundedOrigin = FloatPoint(deviceOrigin.x / deviceScaleX, deviceOrigin.y / deviceScaleY);
    FloatPoint roundedLowerRight = FloatPoint(deviceLowerRight.x / deviceScaleX, deviceLowerRight.y / deviceScaleY);
    return FloatRect(roundedOrigin, roundedLowerRight - roundedOrigin);
}

inline FloatRect cgRoundToDevicePixels(CGAffineTransform deviceMatrix, FloatRect rect)
{
    if (CGAffineTransformIsIdentity(deviceMatrix))
        return roundedIntRect(rect);
    return cgRoundToDevicePixelsNonIdentity(deviceMatrix, rect);
}

inline IntRect cgImageRect(CGImageRef image)
{
    return { 0, 0, static_cast<int>(CGImageGetWidth(image)), static_cast<int>(CGImageGetHeight(image)) };
}

inline std::span<CGPoint> pointsSpan(const CGPathElement* element)
{
    switch (element->type) {
    case kCGPathElementMoveToPoint:
        return unsafeMakeSpan(element->points, 1);
    case kCGPathElementAddLineToPoint:
        return unsafeMakeSpan(element->points, 1);
    case kCGPathElementAddQuadCurveToPoint:
        return unsafeMakeSpan(element->points, 2);
    case kCGPathElementAddCurveToPoint:
        return unsafeMakeSpan(element->points, 3);
    case kCGPathElementCloseSubpath:
        break;
    }
    return { };
}

}
