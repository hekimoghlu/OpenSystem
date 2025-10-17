/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include "RemoteRenderBundle.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteRenderBundleMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPURenderBundle.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderBundle);

RemoteRenderBundle::RemoteRenderBundle(WebCore::WebGPU::RenderBundle& renderBundle, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(renderBundle)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteRenderBundle::messageReceiverName(), m_identifier.toUInt64());
}

RemoteRenderBundle::~RemoteRenderBundle() = default;

void RemoteRenderBundle::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteRenderBundle::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteRenderBundle::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteRenderBundle::setLabel(String&& label)
{
    Ref { m_backing }->setLabel(WTFMove(label));
}

Ref<IPC::StreamServerConnection> RemoteRenderBundle::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
