/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#import "TextDetectorImplementation.h"

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#import "DetectedTextInterface.h"
#import "ImageBuffer.h"
#import "NativeImage.h"
#import "VisionUtilities.h"
#import <wtf/TZoneMallocInlines.h>
#import <pal/cocoa/VisionSoftLink.h>

namespace WebCore::ShapeDetection {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextDetectorImpl);

TextDetectorImpl::TextDetectorImpl() = default;

TextDetectorImpl::~TextDetectorImpl() = default;

void TextDetectorImpl::detect(Ref<ImageBuffer>&& imageBuffer, CompletionHandler<void(Vector<DetectedText>&&)>&& completionHandler)
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

    auto request = adoptNS([PAL::allocVNRecognizeTextRequestInstance() init]);
    configureRequestToUseCPUOrGPU(request.get());

    auto imageRequestHandler = adoptNS([PAL::allocVNImageRequestHandlerInstance() initWithCGImage:platformImage.get() options:@{ }]);

    NSError *error = nil;
    auto result = [imageRequestHandler performRequests:@[request.get()] error:&error];
    if (!result || error) {
        completionHandler({ });
        return;
    }

    Vector<DetectedText> results;
    results.reserveInitialCapacity(request.get().results.count);
    for (VNRecognizedTextObservation *observation in request.get().results) {
        results.append({
            convertRectFromVisionToWeb(nativeImage->size(), observation.boundingBox),
            [observation topCandidates:1][0].string,
            convertCornerPoints(nativeImage->size(), observation),
        });
    }

    completionHandler(WTFMove(results));
}

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
