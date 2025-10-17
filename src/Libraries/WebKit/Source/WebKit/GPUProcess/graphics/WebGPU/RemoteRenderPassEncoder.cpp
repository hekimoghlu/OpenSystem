/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "RemoteRenderPassEncoder.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteRenderPassEncoderMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBindGroup.h>
#include <WebCore/WebGPUBuffer.h>
#include <WebCore/WebGPURenderBundle.h>
#include <WebCore/WebGPURenderPassEncoder.h>
#include <WebCore/WebGPURenderPipeline.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderPassEncoder);

RemoteRenderPassEncoder::RemoteRenderPassEncoder(WebCore::WebGPU::RenderPassEncoder& renderPassEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(renderPassEncoder)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    Ref { m_streamConnection }->startReceivingMessages(*this, Messages::RemoteRenderPassEncoder::messageReceiverName(), m_identifier.toUInt64());
}

RemoteRenderPassEncoder::~RemoteRenderPassEncoder() = default;

void RemoteRenderPassEncoder::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteRenderPassEncoder::stopListeningForIPC()
{
    Ref { m_streamConnection }->stopReceivingMessages(Messages::RemoteRenderPassEncoder::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteRenderPassEncoder::setPipeline(WebGPUIdentifier renderPipeline)
{
    auto convertedRenderPipeline = protectedObjectHeap()->convertRenderPipelineFromBacking(renderPipeline);
    ASSERT(convertedRenderPipeline);
    if (!convertedRenderPipeline)
        return;

    protectedBacking()->setPipeline(*convertedRenderPipeline);
}

void RemoteRenderPassEncoder::setIndexBuffer(WebGPUIdentifier buffer, WebCore::WebGPU::IndexFormat indexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedObjectHeap()->convertBufferFromBacking(buffer);
    ASSERT(convertedBuffer);
    if (!convertedBuffer)
        return;

    protectedBacking()->setIndexBuffer(*convertedBuffer, indexFormat, offset, size);
}

void RemoteRenderPassEncoder::setVertexBuffer(WebCore::WebGPU::Index32 slot, WebGPUIdentifier buffer, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    RefPtr convertedBuffer = protectedObjectHeap()->convertBufferFromBacking(buffer).get();
    ASSERT(convertedBuffer);
    if (!convertedBuffer)
        return;

    protectedBacking()->setVertexBuffer(slot, convertedBuffer.get(), offset, size);
}

void RemoteRenderPassEncoder::unsetVertexBuffer(WebCore::WebGPU::Index32 slot, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    protectedBacking()->setVertexBuffer(slot, nullptr, offset, size);
}

void RemoteRenderPassEncoder::draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    protectedBacking()->draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void RemoteRenderPassEncoder::drawIndexed(WebCore::WebGPU::Size32 indexCount,
    std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstIndex,
    std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
    std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    protectedBacking()->drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
}

void RemoteRenderPassEncoder::drawIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedObjectHeap()->convertBufferFromBacking(indirectBuffer);
    ASSERT(convertedIndirectBuffer);
    if (!convertedIndirectBuffer)
        return;

    protectedBacking()->drawIndirect(*convertedIndirectBuffer, indirectOffset);
}

void RemoteRenderPassEncoder::drawIndexedIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedObjectHeap()->convertBufferFromBacking(indirectBuffer);
    ASSERT(convertedIndirectBuffer);
    if (!convertedIndirectBuffer)
        return;

    protectedBacking()->drawIndexedIndirect(*convertedIndirectBuffer, indirectOffset);
}

void RemoteRenderPassEncoder::setBindGroup(WebCore::WebGPU::Index32 index, WebGPUIdentifier bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets)
{
    auto convertedBindGroup = protectedObjectHeap()->convertBindGroupFromBacking(bindGroup);
    if (!convertedBindGroup)
        return;

    protectedBacking()->setBindGroup(index, *convertedBindGroup, WTFMove(dynamicOffsets));
}

void RemoteRenderPassEncoder::pushDebugGroup(String&& groupLabel)
{
    protectedBacking()->pushDebugGroup(WTFMove(groupLabel));
}

void RemoteRenderPassEncoder::popDebugGroup()
{
    protectedBacking()->popDebugGroup();
}

void RemoteRenderPassEncoder::insertDebugMarker(String&& markerLabel)
{
    protectedBacking()->insertDebugMarker(WTFMove(markerLabel));
}

void RemoteRenderPassEncoder::setViewport(float x, float y,
    float width, float height,
    float minDepth, float maxDepth)
{
    protectedBacking()->setViewport(x, y, width, height, minDepth, maxDepth);
}

void RemoteRenderPassEncoder::setScissorRect(WebCore::WebGPU::IntegerCoordinate x, WebCore::WebGPU::IntegerCoordinate y,
    WebCore::WebGPU::IntegerCoordinate width, WebCore::WebGPU::IntegerCoordinate height)
{
    protectedBacking()->setScissorRect(x, y, width, height);
}

void RemoteRenderPassEncoder::setBlendConstant(WebGPU::Color color)
{
    auto convertedColor = protectedObjectHeap()->convertFromBacking(color);
    ASSERT(convertedColor);
    if (!convertedColor)
        return;

    protectedBacking()->setBlendConstant(*convertedColor);
}

void RemoteRenderPassEncoder::setStencilReference(WebCore::WebGPU::StencilValue stencilValue)
{
    protectedBacking()->setStencilReference(stencilValue);
}

void RemoteRenderPassEncoder::beginOcclusionQuery(WebCore::WebGPU::Size32 queryIndex)
{
    protectedBacking()->beginOcclusionQuery(queryIndex);
}

void RemoteRenderPassEncoder::endOcclusionQuery()
{
    protectedBacking()->endOcclusionQuery();
}

void RemoteRenderPassEncoder::executeBundles(Vector<WebGPUIdentifier>&& renderBundles)
{
    Vector<Ref<WebCore::WebGPU::RenderBundle>> convertedBundles;
    convertedBundles.reserveInitialCapacity(renderBundles.size());
    for (WebGPUIdentifier identifier : renderBundles) {
        auto convertedBundle = protectedObjectHeap()->convertRenderBundleFromBacking(identifier);
        ASSERT(convertedBundle);
        if (!convertedBundle)
            return;
        convertedBundles.append(*convertedBundle);
    }
    protectedBacking()->executeBundles(WTFMove(convertedBundles));
}

void RemoteRenderPassEncoder::end()
{
    protectedBacking()->end();
}

void RemoteRenderPassEncoder::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::RenderPassEncoder> RemoteRenderPassEncoder::protectedBacking()
{
    return m_backing;
}

Ref<WebGPU::ObjectHeap> RemoteRenderPassEncoder::protectedObjectHeap() const
{
    return m_objectHeap.get();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
