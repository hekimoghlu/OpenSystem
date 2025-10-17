/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include "RemoteBarcodeDetectorProxy.h"

#if ENABLE(GPU_PROCESS)

#include "ArgumentCoders.h"
#include "MessageSenderInlines.h"
#include "RemoteBarcodeDetectorMessages.h"
#include "RemoteRenderingBackendProxy.h"
#include "StreamClientConnection.h"
#include "WebProcess.h"
#include <WebCore/ImageBuffer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::ShapeDetection {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteBarcodeDetectorProxy);

Ref<RemoteBarcodeDetectorProxy> RemoteBarcodeDetectorProxy::create(Ref<IPC::StreamClientConnection>&& streamClientConnection, RenderingBackendIdentifier renderingBackendIdentifier, ShapeDetectionIdentifier identifier, const WebCore::ShapeDetection::BarcodeDetectorOptions& barcodeDetectorOptions)
{
    streamClientConnection->send(Messages::RemoteRenderingBackend::CreateRemoteBarcodeDetector(identifier, barcodeDetectorOptions), renderingBackendIdentifier);
    return adoptRef(*new RemoteBarcodeDetectorProxy(WTFMove(streamClientConnection), renderingBackendIdentifier, identifier));
}

RemoteBarcodeDetectorProxy::RemoteBarcodeDetectorProxy(Ref<IPC::StreamClientConnection>&& streamClientConnection, RenderingBackendIdentifier renderingBackendIdentifier, ShapeDetectionIdentifier identifier)
    : m_backing(identifier)
    , m_streamClientConnection(WTFMove(streamClientConnection))
    , m_renderingBackendIdentifier(renderingBackendIdentifier)
{
}

RemoteBarcodeDetectorProxy::~RemoteBarcodeDetectorProxy()
{
    m_streamClientConnection->send(Messages::RemoteRenderingBackend::ReleaseRemoteBarcodeDetector(m_backing), m_renderingBackendIdentifier);
}

void RemoteBarcodeDetectorProxy::getSupportedFormats(Ref<IPC::StreamClientConnection>&& streamClientConnection, RenderingBackendIdentifier renderingBackendIdentifier, CompletionHandler<void(Vector<WebCore::ShapeDetection::BarcodeFormat>&&)>&& completionHandler)
{
    streamClientConnection->sendWithAsyncReply(Messages::RemoteRenderingBackend::GetRemoteBarcodeDetectorSupportedFormats(), WTFMove(completionHandler), renderingBackendIdentifier);
}

void RemoteBarcodeDetectorProxy::detect(Ref<WebCore::ImageBuffer>&& imageBuffer, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedBarcode>&&)>&& completionHandler)
{
    m_streamClientConnection->sendWithAsyncReply(Messages::RemoteBarcodeDetector::Detect(imageBuffer->renderingResourceIdentifier()), WTFMove(completionHandler), m_backing);
}

} // namespace WebKit::WebGPU

#endif // HAVE(GPU_PROCESS)
