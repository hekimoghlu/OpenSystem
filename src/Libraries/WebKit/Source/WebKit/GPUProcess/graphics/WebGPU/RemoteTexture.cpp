/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#include "RemoteTexture.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "Logging.h"
#include "RemoteTextureMessages.h"
#include "RemoteTextureView.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include "WebGPUTextureViewDescriptor.h"
#include <WebCore/WebGPUTexture.h>
#include <WebCore/WebGPUTextureView.h>
#include <WebCore/WebGPUTextureViewDescriptor.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_OPTIONAL_CONNECTION_BASE(assertion, connection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteTexture);

RemoteTexture::RemoteTexture(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::Texture& texture, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(texture)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_identifier(identifier)
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_gpu(gpu)
{
    Ref { m_streamConnection }->startReceivingMessages(*this, Messages::RemoteTexture::messageReceiverName(), m_identifier.toUInt64());
}

RemoteTexture::~RemoteTexture() = default;

RefPtr<IPC::Connection> RemoteTexture::connection() const
{
    RefPtr connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteTexture::stopListeningForIPC()
{
    Ref { m_streamConnection }->stopReceivingMessages(Messages::RemoteTexture::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteTexture::createView(const std::optional<WebGPU::TextureViewDescriptor>& descriptor, WebGPUIdentifier identifier)
{
    std::optional<WebCore::WebGPU::TextureViewDescriptor> convertedDescriptor;
    Ref objectHeap = m_objectHeap.get();

    if (descriptor) {
        auto resultDescriptor = objectHeap->convertFromBacking(*descriptor);
        MESSAGE_CHECK(resultDescriptor);
        convertedDescriptor = WTFMove(resultDescriptor);
    }
    auto textureView = protectedBacking()->createView(convertedDescriptor);
    MESSAGE_CHECK(textureView);
    auto remoteTextureView = RemoteTextureView::create(textureView.releaseNonNull(), objectHeap, m_streamConnection.copyRef(), Ref { m_gpu.get() }, identifier);
    objectHeap->addObject(identifier, remoteTextureView);
}

void RemoteTexture::destroy()
{
    protectedBacking()->destroy();
}

void RemoteTexture::undestroy()
{
    protectedBacking()->undestroy();
}

void RemoteTexture::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteTexture::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::Texture> RemoteTexture::protectedBacking()
{
    return m_backing;
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
