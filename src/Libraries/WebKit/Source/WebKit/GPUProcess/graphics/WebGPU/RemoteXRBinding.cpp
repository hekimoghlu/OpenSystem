/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "RemoteXRBinding.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "RemoteGPU.h"
#include "RemoteXRBindingMessages.h"
#include "RemoteXRProjectionLayer.h"
#include "RemoteXRSubImage.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUXRBinding.h>
#include <WebCore/WebGPUXRProjectionLayer.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRBinding);

RemoteXRBinding::RemoteXRBinding(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::WebGPU::XRBinding& xrBinding, WebGPU::ObjectHeap& objectHeap, RemoteGPU& gpu, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(xrBinding)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_identifier(identifier)
    , m_gpu(gpu)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteXRBinding::messageReceiverName(), m_identifier.toUInt64());
}

RemoteXRBinding::~RemoteXRBinding() = default;

Ref<IPC::StreamServerConnection> RemoteXRBinding::protectedStreamConnection()
{
    return m_streamConnection;
}

Ref<WebCore::WebGPU::XRBinding> RemoteXRBinding::protectedBacking()
{
    return m_backing;
}

Ref<RemoteGPU> RemoteXRBinding::protectedGPU()
{
    return m_gpu.get();
}

void RemoteXRBinding::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteXRBinding::createProjectionLayer(WebCore::WebGPU::TextureFormat colorFormat, std::optional<WebCore::WebGPU::TextureFormat> depthStencilFormat, WebCore::WebGPU::TextureUsageFlags textureUsage, double scaleFactor, WebGPUIdentifier identifier)
{
    WebCore::WebGPU::XRProjectionLayerInit init {
        .colorFormat = colorFormat,
        .depthStencilFormat = WTFMove(depthStencilFormat),
        .textureUsage = textureUsage,
        .scaleFactor = scaleFactor
    };
    RefPtr projectionLayer = protectedBacking()->createProjectionLayer(WTFMove(init));
    if (!projectionLayer) {
        // FIXME: Add MESSAGE_CHECK call
        return;
    }

    Ref objectHeap = m_objectHeap.get();
    Ref remoteProjectionLayer = RemoteXRProjectionLayer::create(*projectionLayer, objectHeap, protectedStreamConnection(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteProjectionLayer);
}

void RemoteXRBinding::getViewSubImage(WebGPUIdentifier projectionLayerIdentifier, WebGPUIdentifier identifier)
{
    Ref objectHeap = m_objectHeap.get();
    auto projectionLayer = objectHeap->convertXRProjectionLayerFromBacking(projectionLayerIdentifier);
    if (!projectionLayer) {
        // FIXME: Add MESSAGE_CHECK call
        return;
    }

    RefPtr subImage = protectedBacking()->getViewSubImage(*projectionLayer);
    if (!subImage) {
        // FIXME: Add MESSAGE_CHECK call
        return;
    }

    Ref remoteSubImage = RemoteXRSubImage::create(*m_gpuConnectionToWebProcess.get(), *subImage, objectHeap, protectedStreamConnection(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteSubImage);
}

void RemoteXRBinding::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteXRBinding::messageReceiverName(), m_identifier.toUInt64());
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
