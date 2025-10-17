/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "RemoteXRSubImage.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "RemoteGPU.h"
#include "RemoteTexture.h"
#include "RemoteXRSubImageMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUTexture.h>
#include <WebCore/WebGPUXRSubImage.h>
#include <wtf/TZoneMalloc.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_OPTIONAL_CONNECTION_BASE(assertion, connection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRSubImage);

RemoteXRSubImage::RemoteXRSubImage(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::WebGPU::XRSubImage& xrSubImage, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(xrSubImage)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_identifier(identifier)
    , m_gpu(gpu)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteXRSubImage::messageReceiverName(), m_identifier.toUInt64());
}

RemoteXRSubImage::~RemoteXRSubImage() = default;

Ref<IPC::StreamServerConnection> RemoteXRSubImage::protectedStreamConnection()
{
    return m_streamConnection;
}

Ref<WebCore::WebGPU::XRSubImage> RemoteXRSubImage::protectedBacking()
{
    return m_backing;
}

Ref<RemoteGPU> RemoteXRSubImage::protectedGPU() const
{
    return m_gpu.get();
}

RefPtr<IPC::Connection> RemoteXRSubImage::connection() const
{
    RefPtr connection = protectedGPU()->gpuConnectionToWebProcess();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteXRSubImage::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteXRSubImage::getColorTexture(WebGPUIdentifier identifier)
{
    auto texture = protectedBacking()->colorTexture();
    ASSERT(texture);
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!texture || !connection)
        return;

    Ref objectHeap = m_objectHeap.get();
    auto remoteTexture = RemoteTexture::create(*connection, protectedGPU(), *texture, objectHeap, protectedStreamConnection(), identifier);
    objectHeap->addObject(identifier, remoteTexture);
}

void RemoteXRSubImage::getDepthTexture(WebGPUIdentifier identifier)
{
    auto texture = protectedBacking()->depthStencilTexture();
    ASSERT(texture);
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!texture || !connection)
        return;

    Ref objectHeap = m_objectHeap.get();
    auto remoteTexture = RemoteTexture::create(*connection, protectedGPU(), *texture, objectHeap, protectedStreamConnection(), identifier);
    objectHeap->addObject(identifier, remoteTexture);
}

void RemoteXRSubImage::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteXRSubImage::messageReceiverName(), m_identifier.toUInt64());
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
