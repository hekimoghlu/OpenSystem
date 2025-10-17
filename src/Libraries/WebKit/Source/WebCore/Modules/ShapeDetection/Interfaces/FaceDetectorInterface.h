/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {
class ImageBuffer;
}

namespace WebCore::ShapeDetection {

struct DetectedFace;

class FaceDetector : public RefCounted<FaceDetector> {
public:
    virtual ~FaceDetector() = default;

    virtual void detect(Ref<ImageBuffer>&&, CompletionHandler<void(Vector<DetectedFace>&&)>&&) = 0;

protected:
    FaceDetector() = default;

private:
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector(FaceDetector&&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;
    FaceDetector& operator=(FaceDetector&&) = delete;
};

} // namespace WebCore::ShapeDetection
