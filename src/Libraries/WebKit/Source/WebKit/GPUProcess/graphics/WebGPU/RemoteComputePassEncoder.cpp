/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "RemoteComputePassEncoder.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteComputePassEncoderMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroup.h>
#include <WebCore/WebGPUBuffer.h>
#include <WebCore/WebGPUComputePassEncoder.h>
#include <WebCore/WebGPUComputePipeline.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteComputePassEncoder);

RemoteComputePassEncoder::RemoteComputePassEncoder(WebCore::WebGPU::ComputePassEncoder& computePassEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(computePassEncoder)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteComputePassEncoder::messageReceiverName(), m_identifier.toUInt64());
}

RemoteComputePassEncoder::~RemoteComputePassEncoder() = default;

void RemoteComputePassEncoder::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteComputePassEncoder::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteComputePassEncoder::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteComputePassEncoder::setPipeline(WebGPUIdentifier computePipeline)
{
    auto convertedComputePipeline = protectedObjectHeap()->convertComputePipelineFromBacking(computePipeline);
    ASSERT(convertedComputePipeline);
    if (!convertedComputePipeline)
        return;

    protectedBacking()->setPipeline(*convertedComputePipeline);
}

void RemoteComputePassEncoder::dispatch(WebCore::WebGPU::Size32 workgroupCountX, WebCore::WebGPU::Size32 workgroupCountY, WebCore::WebGPU::Size32 workgroupCountZ)
{
    protectedBacking()->dispatch(workgroupCountX, workgroupCountY, workgroupCountZ);
}

void RemoteComputePassEncoder::dispatchIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedObjectHeap()->convertBufferFromBacking(indirectBuffer);
    ASSERT(convertedIndirectBuffer);
    if (!convertedIndirectBuffer)
        return;

    protectedBacking()->dispatchIndirect(*convertedIndirectBuffer, indirectOffset);
}

void RemoteComputePassEncoder::end()
{
    protectedBacking()->end();
}

void RemoteComputePassEncoder::setBindGroup(WebCore::WebGPU::Index32 index, WebGPUIdentifier bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& offsets)
{
    auto convertedBindGroup = protectedObjectHeap()->convertBindGroupFromBacking(bindGroup);
    ASSERT(convertedBindGroup);
    if (!convertedBindGroup)
        return;

    protectedBacking()->setBindGroup(index, *convertedBindGroup, WTFMove(offsets));
}

void RemoteComputePassEncoder::pushDebugGroup(String&& groupLabel)
{
    protectedBacking()->pushDebugGroup(WTFMove(groupLabel));
}

void RemoteComputePassEncoder::popDebugGroup()
{
    protectedBacking()->popDebugGroup();
}

void RemoteComputePassEncoder::insertDebugMarker(String&& markerLabel)
{
    protectedBacking()->insertDebugMarker(WTFMove(markerLabel));
}

void RemoteComputePassEncoder::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::ComputePassEncoder> RemoteComputePassEncoder::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteComputePassEncoder::protectedStreamConnection() const
{
    return m_streamConnection;
}

Ref<WebGPU::ObjectHeap> RemoteComputePassEncoder::protectedObjectHeap() const
{
    return m_objectHeap.get();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
