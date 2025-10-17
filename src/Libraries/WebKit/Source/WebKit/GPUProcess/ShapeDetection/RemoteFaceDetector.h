/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

#include "Connection.h"
#include "RemoteRenderingBackend.h"
#include "ShapeDetectionIdentifier.h"
#include "StreamMessageReceiver.h"
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/RenderingResourceIdentifier.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore::ShapeDetection {
struct DetectedFace;
class FaceDetector;
}

namespace WebKit {
class RemoteRenderingBackend;
struct SharedPreferencesForWebProcess;

namespace ShapeDetection {
class ObjectHeap;
}

class RemoteFaceDetector : public IPC::StreamMessageReceiver {
public:
    WTF_MAKE_TZONE_ALLOCATED(RemoteFaceDetector);
public:
    static Ref<RemoteFaceDetector> create(Ref<WebCore::ShapeDetection::FaceDetector>&& faceDetector, ShapeDetection::ObjectHeap& objectHeap, RemoteRenderingBackend& backend, ShapeDetectionIdentifier identifier, WebCore::ProcessIdentifier webProcessIdentifier)
    {
        return adoptRef(*new RemoteFaceDetector(WTFMove(faceDetector), objectHeap, backend, identifier, webProcessIdentifier));
    }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    virtual ~RemoteFaceDetector();

private:
    RemoteFaceDetector(Ref<WebCore::ShapeDetection::FaceDetector>&&, ShapeDetection::ObjectHeap&, RemoteRenderingBackend&, ShapeDetectionIdentifier, WebCore::ProcessIdentifier);

    RemoteFaceDetector(const RemoteFaceDetector&) = delete;
    RemoteFaceDetector(RemoteFaceDetector&&) = delete;
    RemoteFaceDetector& operator=(const RemoteFaceDetector&) = delete;
    RemoteFaceDetector& operator=(RemoteFaceDetector&&) = delete;

    WebCore::ShapeDetection::FaceDetector& backing() { return m_backing; }
    Ref<RemoteRenderingBackend> protectedBackend() const { return m_backend.get(); }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void detect(WebCore::RenderingResourceIdentifier, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedFace>&&)>&&);

    Ref<WebCore::ShapeDetection::FaceDetector> m_backing;
    WeakRef<ShapeDetection::ObjectHeap> m_objectHeap;
    WeakRef<RemoteRenderingBackend> m_backend;
    const ShapeDetectionIdentifier m_identifier;
    const WebCore::ProcessIdentifier m_webProcessIdentifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
