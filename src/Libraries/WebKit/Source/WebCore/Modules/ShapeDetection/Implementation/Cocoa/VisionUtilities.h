/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#import <wtf/Vector.h>

@class VNRectangleObservation;
@class VNRequest;

namespace WebCore {
class FloatPoint;
class FloatRect;
class FloatSize;
}

namespace WebCore::ShapeDetection {

// Vision uses normalized coordinates (0 .. 1 across the image), and
// also uses an increasing-y-goes-up coordinate system.
// The web expects the results are in unnormalized coordinates (0 .. size across the image), and
// an increasing-y-goes-down coordinate system.
// These functions perform the necessary conversions.

FloatRect convertRectFromVisionToWeb(const FloatSize& imageSize, const FloatRect&);
FloatPoint convertPointFromVisionToWeb(const FloatSize& imageSize, const FloatPoint&);
FloatPoint convertPointFromUnnormalizedVisionToWeb(const FloatSize& imageSize, const FloatPoint&);
Vector<FloatPoint> convertCornerPoints(const FloatSize& imageSize, VNRectangleObservation *);

void configureRequestToUseCPUOrGPU(VNRequest *);

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
