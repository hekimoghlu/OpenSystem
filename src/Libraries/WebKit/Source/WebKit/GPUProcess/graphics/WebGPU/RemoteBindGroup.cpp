/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#include "RemoteBindGroup.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBindGroupMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroup.h>
#include <WebCore/WebGPUExternalTexture.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteBindGroup);

RemoteBindGroup::RemoteBindGroup(WebCore::WebGPU::BindGroup& bindGroup, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(bindGroup)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteBindGroup::messageReceiverName(), m_identifier.toUInt64());
}

RemoteBindGroup::~RemoteBindGroup() = default;

void RemoteBindGroup::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteBindGroup::updateExternalTextures(WebGPUIdentifier externalTextureIdentifier, CompletionHandler<void(bool)>&& completion)
{
    if (auto externalTexture = protectedObjectHeap()->convertExternalTextureFromBacking(externalTextureIdentifier); externalTexture.get())
        completion(protectedBacking()->updateExternalTextures(*externalTexture.get()));
    else
        completion(false);
}

void RemoteBindGroup::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteBindGroup::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteBindGroup::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::BindGroup> RemoteBindGroup::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteBindGroup::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
