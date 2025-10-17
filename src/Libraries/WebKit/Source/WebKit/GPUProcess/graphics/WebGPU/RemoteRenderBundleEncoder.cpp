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
#include "RemoteRenderBundleEncoder.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "Logging.h"
#include "RemoteRenderBundle.h"
#include "RemoteRenderBundleEncoderMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroup.h>
#include <WebCore/WebGPUBuffer.h>
#include <WebCore/WebGPURenderBundle.h>
#include <WebCore/WebGPURenderBundleEncoder.h>
#include <WebCore/WebGPURenderPipeline.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_OPTIONAL_CONNECTION_BASE(assertion, connection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderBundleEncoder);

RemoteRenderBundleEncoder::RemoteRenderBundleEncoder(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::RenderBundleEncoder& renderBundleEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(renderBundleEncoder)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_identifier(identifier)
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_gpu(gpu)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteRenderBundleEncoder::messageReceiverName(), m_identifier.toUInt64());
}

RemoteRenderBundleEncoder::~RemoteRenderBundleEncoder() = default;

RefPtr<IPC::Connection> RemoteRenderBundleEncoder::connection() const
{
    RefPtr connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteRenderBundleEncoder::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteRenderBundleEncoder::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteRenderBundleEncoder::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteRenderBundleEncoder::setPipeline(WebGPUIdentifier renderPipeline)
{
    auto convertedRenderPipeline = protectedObjectHeap()->convertRenderPipelineFromBacking(renderPipeline);
    ASSERT(convertedRenderPipeline);
    if (!convertedRenderPipeline)
        return;

    protectedBacking()->setPipeline(*convertedRenderPipeline);
}

void RemoteRenderBundleEncoder::setIndexBuffer(WebGPUIdentifier buffer, WebCore::WebGPU::IndexFormat indexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    RefPtr convertedBuffer = protectedObjectHeap()->convertBufferFromBacking(buffer).get();
    ASSERT(convertedBuffer);
    if (!convertedBuffer)
        return;

    protectedBacking()->setIndexBuffer(*convertedBuffer, indexFormat, offset, size);
}

void RemoteRenderBundleEncoder::setVertexBuffer(WebCore::WebGPU::Index32 slot, WebGPUIdentifier buffer, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    RefPtr convertedBuffer = protectedObjectHeap()->convertBufferFromBacking(buffer).get();
    ASSERT(convertedBuffer);
    if (!convertedBuffer)
        return;

    protectedBacking()->setVertexBuffer(slot, convertedBuffer.get(), offset, size);
}

void RemoteRenderBundleEncoder::unsetVertexBuffer(WebCore::WebGPU::Index32 slot, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    protectedBacking()->setVertexBuffer(slot, nullptr, offset, size);
}

void RemoteRenderBundleEncoder::draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    protectedBacking()->draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void RemoteRenderBundleEncoder::drawIndexed(WebCore::WebGPU::Size32 indexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstIndex,
    std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
    std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    protectedBacking()->drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
}

void RemoteRenderBundleEncoder::drawIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    RefPtr convertedIndirectBuffer = protectedObjectHeap()->convertBufferFromBacking(indirectBuffer).get();
    ASSERT(convertedIndirectBuffer);
    if (!convertedIndirectBuffer)
        return;

    protectedBacking()->drawIndirect(*convertedIndirectBuffer, indirectOffset);
}

void RemoteRenderBundleEncoder::drawIndexedIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    RefPtr convertedIndirectBuffer = protectedObjectHeap()->convertBufferFromBacking(indirectBuffer).get();
    ASSERT(convertedIndirectBuffer);
    if (!convertedIndirectBuffer)
        return;

    protectedBacking()->drawIndexedIndirect(*convertedIndirectBuffer, indirectOffset);
}

void RemoteRenderBundleEncoder::setBindGroup(WebCore::WebGPU::Index32 index, WebGPUIdentifier bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets)
{
    auto convertedBindGroup = protectedObjectHeap()->convertBindGroupFromBacking(bindGroup);
    ASSERT(convertedBindGroup);
    if (!convertedBindGroup)
        return;

    protectedBacking()->setBindGroup(index, *convertedBindGroup, WTFMove(dynamicOffsets));
}

void RemoteRenderBundleEncoder::pushDebugGroup(String&& groupLabel)
{
    protectedBacking()->pushDebugGroup(WTFMove(groupLabel));
}

void RemoteRenderBundleEncoder::popDebugGroup()
{
    protectedBacking()->popDebugGroup();
}

void RemoteRenderBundleEncoder::insertDebugMarker(String&& markerLabel)
{
    protectedBacking()->insertDebugMarker(WTFMove(markerLabel));
}

void RemoteRenderBundleEncoder::finish(const WebGPU::RenderBundleDescriptor& descriptor, WebGPUIdentifier identifier)
{
    Ref objectHeap = m_objectHeap.get();
    auto convertedDescriptor = objectHeap->convertFromBacking(descriptor);
    MESSAGE_CHECK(convertedDescriptor);

    auto renderBundle = protectedBacking()->finish(*convertedDescriptor);
    MESSAGE_CHECK(renderBundle);
    auto remoteRenderBundle = RemoteRenderBundle::create(*renderBundle, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteRenderBundle);
}

void RemoteRenderBundleEncoder::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::RenderBundleEncoder> RemoteRenderBundleEncoder::protectedBacking() const
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteRenderBundleEncoder::protectedStreamConnection() const
{
    return m_streamConnection;
}

Ref<WebGPU::ObjectHeap> RemoteRenderBundleEncoder::protectedObjectHeap() const
{
    return m_objectHeap.get();
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
