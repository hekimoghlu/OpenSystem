/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "FaceDetector.h"

#include "Chrome.h"
#include "DetectedFace.h"
#include "Document.h"
#include "FaceDetectorOptions.h"
#include "ImageBitmap.h"
#include "ImageBitmapOptions.h"
#include "ImageBuffer.h"
#include "JSDOMPromiseDeferred.h"
#include "JSDetectedFace.h"
#include "Page.h"
#include "ScriptExecutionContext.h"
#include "WorkerGlobalScope.h"

namespace WebCore {

ExceptionOr<Ref<FaceDetector>> FaceDetector::create(ScriptExecutionContext& scriptExecutionContext, const FaceDetectorOptions& faceDetectorOptions)
{
    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext)) {
        RefPtr page = document->page();
        if (!page)
            return Exception { ExceptionCode::AbortError };
        auto backing = page->chrome().createFaceDetector(faceDetectorOptions.convertToBacking());
        if (!backing)
            return Exception { ExceptionCode::AbortError };
        return adoptRef(*new FaceDetector(backing.releaseNonNull()));
    }

    if (is<WorkerGlobalScope>(scriptExecutionContext)) {
        // FIXME: https://bugs.webkit.org/show_bug.cgi?id=255380 Make the Shape Detection API work in Workers
        return Exception { ExceptionCode::AbortError };
    }

    return Exception { ExceptionCode::AbortError };
}

FaceDetector::FaceDetector(Ref<ShapeDetection::FaceDetector>&& backing)
    : m_backing(WTFMove(backing))
{
}

FaceDetector::~FaceDetector() = default;

void FaceDetector::detect(ScriptExecutionContext& scriptExecutionContext, ImageBitmap::Source&& source, DetectPromise&& promise)
{
    ImageBitmap::createCompletionHandler(scriptExecutionContext, WTFMove(source), { }, [backing = m_backing.copyRef(), promise = WTFMove(promise)](ExceptionOr<Ref<ImageBitmap>>&& imageBitmap) mutable {
        if (imageBitmap.hasException()) {
            promise.resolve({ });
            return;
        }

        auto imageBuffer = imageBitmap.releaseReturnValue()->takeImageBuffer();
        if (!imageBuffer) {
            promise.resolve({ });
            return;
        }

        backing->detect(imageBuffer.releaseNonNull(), [promise = WTFMove(promise)](Vector<ShapeDetection::DetectedFace>&& detectedFaces) mutable {
            promise.resolve(detectedFaces.map([](const auto& detectedFace) {
                return convertFromBacking(detectedFace);
            }));
        });
    });
}

} // namespace WebCore
