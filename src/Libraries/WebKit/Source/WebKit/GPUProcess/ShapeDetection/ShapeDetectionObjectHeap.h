/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "ScopedActiveMessageReceiveQueue.h"
#include "ShapeDetectionIdentifier.h"
#include <functional>
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::ShapeDetection {
class BarcodeDetector;
class FaceDetector;
class TextDetector;
}

namespace WebKit {
class RemoteBarcodeDetector;
class RemoteFaceDetector;
class RemoteTextDetector;
}

namespace WebKit::ShapeDetection {

class ObjectHeap final : public RefCountedAndCanMakeWeakPtr<ObjectHeap> {
    WTF_MAKE_TZONE_ALLOCATED(ObjectHeap);
public:
    static Ref<ObjectHeap> create()
    {
        return adoptRef(*new ObjectHeap);
    }

    ~ObjectHeap();

    void addObject(ShapeDetectionIdentifier, RemoteBarcodeDetector&);
    void addObject(ShapeDetectionIdentifier, RemoteFaceDetector&);
    void addObject(ShapeDetectionIdentifier, RemoteTextDetector&);

    void removeObject(ShapeDetectionIdentifier);

    void clear();

private:
    ObjectHeap();

    HashMap<ShapeDetectionIdentifier, Ref<RemoteBarcodeDetector>> m_barcodeDetectors;
    HashMap<ShapeDetectionIdentifier, Ref<RemoteFaceDetector>> m_faceDetectors;
    HashMap<ShapeDetectionIdentifier, Ref<RemoteTextDetector>> m_textDetectors;
};

} // namespace WebKit::ShapeDetection

#endif // ENABLE(GPU_PROCESS)
