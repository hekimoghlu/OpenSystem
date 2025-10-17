/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#import "config.h"
#import "VisionUtilities.h"

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#import "FloatPoint.h"
#import "FloatRect.h"
#import "FloatSize.h"
#import <CoreML/CoreML.h>
#import <pal/cocoa/CoreMLSoftLink.h>
#import <pal/cocoa/VisionSoftLink.h>

namespace WebCore::ShapeDetection {

FloatRect convertRectFromVisionToWeb(const FloatSize& imageSize, const FloatRect& rect)
{
    auto x = rect.x() * imageSize.width();
    auto y = rect.y() * imageSize.height();
    auto maxX = rect.maxX() * imageSize.width();
    auto maxY = rect.maxY() * imageSize.height();
    return { x, imageSize.height() - maxY, maxX - x, maxY - y };
}

FloatPoint convertPointFromVisionToWeb(const FloatSize& imageSize, const FloatPoint& point)
{
    return { point.x() * imageSize.width(), imageSize.height() - point.y() * imageSize.height() };
}

FloatPoint convertPointFromUnnormalizedVisionToWeb(const FloatSize& imageSize, const FloatPoint& point)
{
    return { point.x(), imageSize.height() - point.y() };
}

Vector<FloatPoint> convertCornerPoints(const FloatSize& imageSize, VNRectangleObservation *observation)
{
    auto bottomLeft = convertPointFromVisionToWeb(imageSize, observation.bottomLeft);
    auto bottomRight = convertPointFromVisionToWeb(imageSize, observation.bottomRight);
    auto topLeft = convertPointFromVisionToWeb(imageSize, observation.topLeft);
    auto topRight = convertPointFromVisionToWeb(imageSize, observation.topRight);

    // The spec says "in clockwise direction and starting with top-left"
    return { topLeft, topRight, bottomRight, bottomLeft };
}

void configureRequestToUseCPUOrGPU(VNRequest *request)
{
#if USE(VISION_CPU_ONLY_PROPERTY)
    request.usesCPUOnly = YES;
#else
    NSError *error = nil;
    auto *supportedComputeStageDevices = [request supportedComputeStageDevicesAndReturnError:&error];
    if (!supportedComputeStageDevices || error)
        return;

    for (VNComputeStage computeStage in supportedComputeStageDevices) {
        bool set = false;
        for (id<MLComputeDeviceProtocol> device in supportedComputeStageDevices[computeStage]) {
            if ([device isKindOfClass:PAL::getMLGPUComputeDeviceClass()]) {
                [request setComputeDevice:device forComputeStage:computeStage];
                set = true;
                break;
            }
        }
        if (!set) {
            for (id<MLComputeDeviceProtocol> device in supportedComputeStageDevices[computeStage]) {
                if ([device isKindOfClass:PAL::getMLGPUComputeDeviceClass()]) {
                    [request setComputeDevice:device forComputeStage:computeStage];
                    break;
                }
            }
        }
    }
#endif
}

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
