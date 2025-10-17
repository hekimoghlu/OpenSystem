/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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

#include "RenderingBackendIdentifier.h"
#include "ShapeDetectionIdentifier.h"
#include <WebCore/TextDetectorInterface.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if ENABLE(GPU_PROCESS)

namespace IPC {
class StreamClientConnection;
}

namespace WebCore::ShapeDetection {
struct DetectedText;
}

namespace WebKit::ShapeDetection {

class RemoteTextDetectorProxy : public WebCore::ShapeDetection::TextDetector {
    WTF_MAKE_TZONE_ALLOCATED(RemoteTextDetectorProxy);
public:
    static Ref<RemoteTextDetectorProxy> create(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, ShapeDetectionIdentifier);

    virtual ~RemoteTextDetectorProxy();

private:
    RemoteTextDetectorProxy(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, ShapeDetectionIdentifier);

    RemoteTextDetectorProxy(const RemoteTextDetectorProxy&) = delete;
    RemoteTextDetectorProxy(RemoteTextDetectorProxy&&) = delete;
    RemoteTextDetectorProxy& operator=(const RemoteTextDetectorProxy&) = delete;
    RemoteTextDetectorProxy& operator=(RemoteTextDetectorProxy&&) = delete;

    ShapeDetectionIdentifier backing() const { return m_backing; }

    void detect(Ref<WebCore::ImageBuffer>&&, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedText>&&)>&&) final;

    ShapeDetectionIdentifier m_backing;
    Ref<IPC::StreamClientConnection> m_streamClientConnection;
    RenderingBackendIdentifier m_renderingBackendIdentifier;
};

} // namespace WebKit::ShapeDetection

#endif // ENABLE(GPU_PROCESS)
