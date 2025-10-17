/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#include "RemoteCompositorIntegration.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "RemoteCompositorIntegrationMessages.h"
#include "RemoteGPU.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUCompositorIntegration.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK_COMPLETION(assertion, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, *connection(), completion)

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteCompositorIntegration);

RemoteCompositorIntegration::RemoteCompositorIntegration(WebCore::WebGPU::CompositorIntegration& compositorIntegration, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(compositorIntegration)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteCompositorIntegration::messageReceiverName(), m_identifier.toUInt64());
}

RemoteCompositorIntegration::~RemoteCompositorIntegration() = default;

RefPtr<IPC::Connection> RemoteCompositorIntegration::connection() const
{
    RefPtr connection = protectedGPU()->gpuConnectionToWebProcess();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteCompositorIntegration::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteCompositorIntegration::paintCompositedResultsToCanvas(WebCore::RenderingResourceIdentifier imageBufferIdentifier, uint32_t bufferIndex, CompletionHandler<void()>&& completionHandler)
{
    UNUSED_PARAM(imageBufferIdentifier);
    protectedBacking()->withDisplayBufferAsNativeImage(bufferIndex, [gpu = m_gpu, imageBufferIdentifier, completionHandler = WTFMove(completionHandler)] (WebCore::NativeImage* image) mutable {
        if (image && gpu.ptr())
            gpu->paintNativeImageToImageBuffer(*image, imageBufferIdentifier);
        completionHandler();
    });
}

void RemoteCompositorIntegration::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteCompositorIntegration::messageReceiverName(), m_identifier.toUInt64());
}

#if PLATFORM(COCOA)
void RemoteCompositorIntegration::recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&& destinationColorSpace, WebCore::AlphaPremultiplication alphaMode, WebCore::WebGPU::TextureFormat textureFormat, WebKit::WebGPUIdentifier deviceIdentifier, CompletionHandler<void(Vector<MachSendRight>&&)>&& callback)
{
    auto convertedDevice = protectedObjectHeap()->convertDeviceFromBacking(deviceIdentifier);
    MESSAGE_CHECK_COMPLETION(convertedDevice, callback({ }));

    callback(protectedBacking()->recreateRenderBuffers(width, height, WTFMove(destinationColorSpace), alphaMode, textureFormat, *convertedDevice));
}
#endif

void RemoteCompositorIntegration::prepareForDisplay(uint32_t frameIndex, CompletionHandler<void(bool)>&& completionHandler)
{
    protectedBacking()->prepareForDisplay(frameIndex, [completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(true);
    });
}

Ref<WebCore::WebGPU::CompositorIntegration> RemoteCompositorIntegration::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteCompositorIntegration::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
