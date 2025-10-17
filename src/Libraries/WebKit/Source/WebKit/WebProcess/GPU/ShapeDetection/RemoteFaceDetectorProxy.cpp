/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "RemoteFaceDetectorProxy.h"

#if ENABLE(GPU_PROCESS)

#include "ArgumentCoders.h"
#include "MessageSenderInlines.h"
#include "RemoteFaceDetectorMessages.h"
#include "RemoteRenderingBackendProxy.h"
#include "StreamClientConnection.h"
#include "WebProcess.h"
#include <WebCore/ImageBuffer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::ShapeDetection {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteFaceDetectorProxy);

Ref<RemoteFaceDetectorProxy> RemoteFaceDetectorProxy::create(Ref<IPC::StreamClientConnection>&& streamClientConnection, RenderingBackendIdentifier renderingBackendIdentifier, ShapeDetectionIdentifier identifier, const WebCore::ShapeDetection::FaceDetectorOptions& faceDetectorOptions)
{
    streamClientConnection->send(Messages::RemoteRenderingBackend::CreateRemoteFaceDetector(identifier, faceDetectorOptions), renderingBackendIdentifier);
    return adoptRef(*new RemoteFaceDetectorProxy(WTFMove(streamClientConnection), renderingBackendIdentifier, identifier));
}

RemoteFaceDetectorProxy::RemoteFaceDetectorProxy(Ref<IPC::StreamClientConnection>&& streamClientConnection, RenderingBackendIdentifier renderingBackendIdentifier, ShapeDetectionIdentifier identifier)
    : m_backing(identifier)
    , m_streamClientConnection(WTFMove(streamClientConnection))
    , m_renderingBackendIdentifier(renderingBackendIdentifier)
{
}

RemoteFaceDetectorProxy::~RemoteFaceDetectorProxy()
{
    protectedStreamClientConnection()->send(Messages::RemoteRenderingBackend::ReleaseRemoteFaceDetector(m_backing), m_renderingBackendIdentifier);
}

void RemoteFaceDetectorProxy::detect(Ref<WebCore::ImageBuffer>&& imageBuffer, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedFace>&&)>&& completionHandler)
{
    protectedStreamClientConnection()->sendWithAsyncReply(Messages::RemoteFaceDetector::Detect(imageBuffer->renderingResourceIdentifier()), WTFMove(completionHandler), m_backing);
}

Ref<IPC::StreamClientConnection> RemoteFaceDetectorProxy::protectedStreamClientConnection() const
{
    return m_streamClientConnection;
}

} // namespace WebKit::WebGPU

#endif // HAVE(GPU_PROCESS)
