/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#include "TextDetectorInterface.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore::ShapeDetection {

class TextDetectorImpl final : public TextDetector {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(TextDetectorImpl, WEBCORE_EXPORT);
public:
    static Ref<TextDetectorImpl> create()
    {
        return adoptRef(*new TextDetectorImpl);
    }

    virtual ~TextDetectorImpl();

private:
    WEBCORE_EXPORT TextDetectorImpl();

    TextDetectorImpl(const TextDetectorImpl&) = delete;
    TextDetectorImpl(TextDetectorImpl&&) = delete;
    TextDetectorImpl& operator=(const TextDetectorImpl&) = delete;
    TextDetectorImpl& operator=(TextDetectorImpl&&) = delete;

    void detect(Ref<ImageBuffer>&&, CompletionHandler<void(Vector<DetectedText>&&)>&&) final;
};

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
