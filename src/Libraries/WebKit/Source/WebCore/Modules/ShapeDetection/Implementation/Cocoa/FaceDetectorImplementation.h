/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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

#include "FaceDetectorInterface.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore::ShapeDetection {

struct FaceDetectorOptions;

class FaceDetectorImpl final : public FaceDetector {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FaceDetectorImpl, WEBCORE_EXPORT);
public:
    static Ref<FaceDetectorImpl> create(const FaceDetectorOptions& faceDetectorOptions)
    {
        return adoptRef(*new FaceDetectorImpl(faceDetectorOptions));
    }

    virtual ~FaceDetectorImpl();

private:
    WEBCORE_EXPORT FaceDetectorImpl(const FaceDetectorOptions&);

    FaceDetectorImpl(const FaceDetectorImpl&) = delete;
    FaceDetectorImpl(FaceDetectorImpl&&) = delete;
    FaceDetectorImpl& operator=(const FaceDetectorImpl&) = delete;
    FaceDetectorImpl& operator=(FaceDetectorImpl&&) = delete;

    void detect(Ref<ImageBuffer>&&, CompletionHandler<void(Vector<DetectedFace>&&)>&&) final;

    uint16_t m_maxDetectedFaces { std::numeric_limits<uint16_t>::max() };
};

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
