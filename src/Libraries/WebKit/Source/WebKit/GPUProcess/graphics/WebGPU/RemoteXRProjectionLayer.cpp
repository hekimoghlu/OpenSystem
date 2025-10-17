/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include "RemoteXRProjectionLayer.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "RemoteGPU.h"
#include "RemoteXRProjectionLayerMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/PlatformXR.h>
#include <WebCore/WebGPUXRProjectionLayer.h>
#include <wtf/MachSendRight.h>
#include <wtf/TZoneMalloc.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_OPTIONAL_CONNECTION_BASE(assertion, connection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRProjectionLayer);

RemoteXRProjectionLayer::RemoteXRProjectionLayer(WebCore::WebGPU::XRProjectionLayer& xrProjectionLayer, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(xrProjectionLayer)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_identifier(identifier)
    , m_gpu(gpu)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteXRProjectionLayer::messageReceiverName(), m_identifier.toUInt64());
}

RemoteXRProjectionLayer::~RemoteXRProjectionLayer() = default;

Ref<WebCore::WebGPU::XRProjectionLayer> RemoteXRProjectionLayer::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteXRProjectionLayer::protectedStreamConnection()
{
    return m_streamConnection;
}

Ref<RemoteGPU> RemoteXRProjectionLayer::protectedGPU() const
{
    return m_gpu.get();
}

RefPtr<IPC::Connection> RemoteXRProjectionLayer::connection() const
{
    RefPtr connection = protectedGPU()->gpuConnectionToWebProcess();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteXRProjectionLayer::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

#if PLATFORM(COCOA)
void RemoteXRProjectionLayer::startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex)
{
    protectedBacking()->startFrame(frameIndex, WTFMove(colorBuffer), WTFMove(depthBuffer), WTFMove(completionSyncEvent), reusableTextureIndex);
}
#endif

void RemoteXRProjectionLayer::endFrame()
{
}

void RemoteXRProjectionLayer::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteXRProjectionLayer::messageReceiverName(), m_identifier.toUInt64());
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
