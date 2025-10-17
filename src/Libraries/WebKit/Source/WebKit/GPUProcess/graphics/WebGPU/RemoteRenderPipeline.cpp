/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#include "RemoteRenderPipeline.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBindGroupLayout.h"
#include "RemoteRenderPipelineMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroupLayout.h>
#include <WebCore/WebGPURenderPipeline.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderPipeline);

RemoteRenderPipeline::RemoteRenderPipeline(WebCore::WebGPU::RenderPipeline& renderPipeline, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(renderPipeline)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteRenderPipeline::messageReceiverName(), m_identifier.toUInt64());
}

RemoteRenderPipeline::~RemoteRenderPipeline() = default;

void RemoteRenderPipeline::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteRenderPipeline::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteRenderPipeline::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteRenderPipeline::getBindGroupLayout(uint32_t index, WebGPUIdentifier identifier)
{
    // "A new GPUBindGroupLayout wrapper is returned each time"
    auto bindGroupLayout = protectedBacking()->getBindGroupLayout(index);
    Ref objectHeap = m_objectHeap.get();
    auto remoteBindGroupLayout = RemoteBindGroupLayout::create(bindGroupLayout, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteBindGroupLayout);
}

void RemoteRenderPipeline::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::RenderPipeline> RemoteRenderPipeline::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteRenderPipeline::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
