/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#include "RemotePresentationContext.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "RemotePresentationContextMessages.h"
#include "RemoteTexture.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUCanvasConfiguration.h>
#include <WebCore/WebGPUPresentationContext.h>
#include <WebCore/WebGPUTexture.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemotePresentationContext);

RemotePresentationContext::RemotePresentationContext(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::PresentationContext& presentationContext, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(presentationContext)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_identifier(identifier)
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_gpu(gpu)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemotePresentationContext::messageReceiverName(), m_identifier.toUInt64());
}

RemotePresentationContext::~RemotePresentationContext() = default;

void RemotePresentationContext::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemotePresentationContext::messageReceiverName(), m_identifier.toUInt64());
}

void RemotePresentationContext::configure(const WebGPU::CanvasConfiguration& canvasConfiguration)
{
    auto convertedConfiguration = m_objectHeap->convertFromBacking(canvasConfiguration);
    ASSERT(convertedConfiguration);
    if (!convertedConfiguration)
        return;

    bool success = protectedBacking()->configure(*convertedConfiguration);
    ASSERT_UNUSED(success, success);
}

void RemotePresentationContext::unconfigure()
{
    protectedBacking()->unconfigure();
}

void RemotePresentationContext::present(uint32_t frameIndex)
{
    protectedBacking()->present(frameIndex);
}

void RemotePresentationContext::getCurrentTexture(WebGPUIdentifier identifier, uint32_t frameIndex)
{
    auto texture = protectedBacking()->getCurrentTexture(frameIndex);
    ASSERT(texture);
    auto connection = m_gpuConnectionToWebProcess.get();
    if (!texture || !connection)
        return;

    // We're creating a new resource here, because we don't want the GetCurrentTexture message to be sync.
    // If the message is async, then the WebGPUIdentifier goes from the Web process to the GPU Process, which
    // means the Web Process is going to proceed and interact with the texture as-if it has this identifier.
    // So we need to make sure the texture has this identifier.
    // Maybe one day we could add the same texture into the ObjectHeap multiple times under multiple identifiers,
    // but for now let's just create a new RemoteTexture object with the expected identifier, just for simplicity.
    // The Web Process should already be caching these current textures internally, so it's unlikely that we'll
    // actually run into a problem here.
    Ref objectHeap = m_objectHeap.get();
    auto remoteTexture = RemoteTexture::create(*connection, protectedGPU(), *texture, objectHeap, m_streamConnection.copyRef(), identifier);
    objectHeap->addObject(identifier, remoteTexture);
}

Ref<WebCore::WebGPU::PresentationContext> RemotePresentationContext::protectedBacking()
{
    return m_backing;
}

Ref<WebGPU::ObjectHeap> RemotePresentationContext::protectedObjectHeap() const
{
    return m_objectHeap.get();
}

Ref<IPC::StreamServerConnection> RemotePresentationContext::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
