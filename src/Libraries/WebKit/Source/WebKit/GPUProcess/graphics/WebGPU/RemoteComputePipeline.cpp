/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#include "RemoteComputePipeline.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBindGroupLayout.h"
#include "RemoteComputePipelineMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroupLayout.h>
#include <WebCore/WebGPUComputePipeline.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteComputePipeline);

RemoteComputePipeline::RemoteComputePipeline(WebCore::WebGPU::ComputePipeline& computePipeline, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(computePipeline)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteComputePipeline::messageReceiverName(), m_identifier.toUInt64());
}

RemoteComputePipeline::~RemoteComputePipeline() = default;

void RemoteComputePipeline::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteComputePipeline::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteComputePipeline::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteComputePipeline::getBindGroupLayout(uint32_t index, WebGPUIdentifier identifier)
{
    // "A new GPUBindGroupLayout wrapper is returned each time"
    Ref objectHeap = m_objectHeap.get();
    auto bindGroupLayout = protectedBacking()->getBindGroupLayout(index);
    auto remoteBindGroupLayout = RemoteBindGroupLayout::create(bindGroupLayout, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteBindGroupLayout);
}

void RemoteComputePipeline::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::ComputePipeline> RemoteComputePipeline::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteComputePipeline::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
