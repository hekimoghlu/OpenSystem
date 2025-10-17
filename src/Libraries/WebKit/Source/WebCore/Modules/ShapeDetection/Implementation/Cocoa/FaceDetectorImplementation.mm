/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#import "FaceDetectorImplementation.h"

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#import "DetectedFaceInterface.h"
#import "FaceDetectorOptionsInterface.h"
#import "ImageBuffer.h"
#import "LandmarkInterface.h"
#import "NativeImage.h"
#import "VisionUtilities.h"
#import <wtf/TZoneMallocInlines.h>
#import <pal/cocoa/VisionSoftLink.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WebCore::ShapeDetection {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FaceDetectorImpl);

FaceDetectorImpl::FaceDetectorImpl(const FaceDetectorOptions& faceDetectorOptions)
    : m_maxDetectedFaces(faceDetectorOptions.maxDetectedFaces)
{
}

FaceDetectorImpl::~FaceDetectorImpl() = default;

static Vector<FloatPoint> convertLandmark(VNFaceLandmarkRegion2D *landmark, const FloatSize& imageSize)
{
    return Vector(std::span { [landmark pointsInImageOfSize:imageSize], landmark.pointCount }).map([&imageSize](const CGPoint& point) {
        return convertPointFromUnnormalizedVisionToWeb(imageSize, point);
    });
}

static Vector<Landmark> convertLandmarks(VNFaceLandmarks2D *landmarks, const FloatSize& imageSize)
{
    return {
        {
            convertLandmark(landmarks.leftEye, imageSize),
            WebCore::ShapeDetection::LandmarkType::Eye,
        }, {
            convertLandmark(landmarks.rightEye, imageSize),
            WebCore::ShapeDetection::LandmarkType::Eye,
        }, {
            convertLandmark(landmarks.nose, imageSize),
            WebCore::ShapeDetection::LandmarkType::Nose,
        },
    };
}

void FaceDetectorImpl::detect(Ref<ImageBuffer>&& imageBuffer, CompletionHandler<void(Vector<DetectedFace>&&)>&& completionHandler)
{
    auto nativeImage = imageBuffer->copyNativeImage();
    if (!nativeImage) {
        completionHandler({ });
        return;
    }

    auto platformImage = nativeImage->platformImage();
    if (!platformImage) {
        completionHandler({ });
        return;
    }

    auto request = adoptNS([PAL::allocVNDetectFaceLandmarksRequestInstance() init]);
    configureRequestToUseCPUOrGPU(request.get());

    auto imageRequestHandler = adoptNS([PAL::allocVNImageRequestHandlerInstance() initWithCGImage:platformImage.get() options:@{ }]);

    NSError *error = nil;
    auto result = [imageRequestHandler performRequests:@[request.get()] error:&error];
    if (!result || error) {
        completionHandler({ });
        return;
    }

    Vector<DetectedFace> results;
    results.reserveInitialCapacity(std::min<size_t>(m_maxDetectedFaces, request.get().results.count));
    for (VNFaceObservation *observation in request.get().results) {
        results.append({
            convertRectFromVisionToWeb(nativeImage->size(), observation.boundingBox),
            { convertLandmarks(observation.landmarks, nativeImage->size()) },
        });
        if (results.size() >= m_maxDetectedFaces)
            break;
    }

    completionHandler(WTFMove(results));
}

} // namespace WebCore::ShapeDetection

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
