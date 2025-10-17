/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "GPURenderPassEncoder.h"

#include "GPUBindGroup.h"
#include "GPUBuffer.h"
#include "GPUQuerySet.h"
#include "GPURenderBundle.h"
#include "GPURenderPipeline.h"
#include "WebGPUDevice.h"

namespace WebCore {

GPURenderPassEncoder::GPURenderPassEncoder(Ref<WebGPU::RenderPassEncoder>&& backing, WebGPU::Device& device)
    : m_backing(WTFMove(backing))
    , m_device(&device)
{
}

String GPURenderPassEncoder::label() const
{
    return m_backing->label();
}

void GPURenderPassEncoder::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

void GPURenderPassEncoder::setPipeline(const GPURenderPipeline& renderPipeline)
{
    m_backing->setPipeline(renderPipeline.backing());
}

void GPURenderPassEncoder::setIndexBuffer(const GPUBuffer& buffer, GPUIndexFormat indexFormat, std::optional<GPUSize64> offset, std::optional<GPUSize64> size)
{
    m_backing->setIndexBuffer(buffer.backing(), convertToBacking(indexFormat), offset, size);
}

void GPURenderPassEncoder::setVertexBuffer(GPUIndex32 slot, const GPUBuffer* buffer, std::optional<GPUSize64> offset, std::optional<GPUSize64> size)
{
    m_backing->setVertexBuffer(slot, buffer ? &buffer->backing() : nullptr, offset, size);
}

void GPURenderPassEncoder::draw(GPUSize32 vertexCount, std::optional<GPUSize32> instanceCount,
    std::optional<GPUSize32> firstVertex, std::optional<GPUSize32> firstInstance)
{
    m_backing->draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void GPURenderPassEncoder::drawIndexed(GPUSize32 indexCount, std::optional<GPUSize32> instanceCount,
    std::optional<GPUSize32> firstIndex,
    std::optional<GPUSignedOffset32> baseVertex,
    std::optional<GPUSize32> firstInstance)
{
    m_backing->drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
}

void GPURenderPassEncoder::drawIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset)
{
    m_backing->drawIndirect(indirectBuffer.backing(), indirectOffset);
}

void GPURenderPassEncoder::drawIndexedIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset)
{
    m_backing->drawIndexedIndirect(indirectBuffer.backing(), indirectOffset);
}

void GPURenderPassEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
    std::optional<Vector<GPUBufferDynamicOffset>>&& dynamicOffsets)
{
    m_backing->setBindGroup(index, bindGroup.backing(), WTFMove(dynamicOffsets));
}

ExceptionOr<void> GPURenderPassEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
    const Uint32Array& dynamicOffsetsData,
    GPUSize64 dynamicOffsetsDataStart,
    GPUSize32 dynamicOffsetsDataLength)
{
    auto offset = checkedSum<uint64_t>(dynamicOffsetsDataStart, dynamicOffsetsDataLength);
    if (offset.hasOverflowed() || offset > dynamicOffsetsData.length())
        return Exception { ExceptionCode::RangeError, "dynamic offsets overflowed"_s };

    m_backing->setBindGroup(index, bindGroup.backing(), dynamicOffsetsData.typedSpan(), dynamicOffsetsDataStart, dynamicOffsetsDataLength);
    return { };
}

void GPURenderPassEncoder::pushDebugGroup(String&& groupLabel)
{
    m_backing->pushDebugGroup(WTFMove(groupLabel));
}

void GPURenderPassEncoder::popDebugGroup()
{
    m_backing->popDebugGroup();
}

void GPURenderPassEncoder::insertDebugMarker(String&& markerLabel)
{
    m_backing->insertDebugMarker(WTFMove(markerLabel));
}

void GPURenderPassEncoder::setViewport(float x, float y,
    float width, float height,
    float minDepth, float maxDepth)
{
    m_backing->setViewport(x, y, width, height, minDepth, maxDepth);
}

void GPURenderPassEncoder::setScissorRect(GPUIntegerCoordinate x, GPUIntegerCoordinate y,
    GPUIntegerCoordinate width, GPUIntegerCoordinate height)
{
    m_backing->setScissorRect(x, y, width, height);
}

void GPURenderPassEncoder::setBlendConstant(GPUColor color)
{
    m_backing->setBlendConstant(convertToBacking(color));
}

void GPURenderPassEncoder::setStencilReference(GPUStencilValue stencilValue)
{
    m_backing->setStencilReference(stencilValue);
}

void GPURenderPassEncoder::beginOcclusionQuery(GPUSize32 queryIndex)
{
    m_backing->beginOcclusionQuery(queryIndex);
}

void GPURenderPassEncoder::endOcclusionQuery()
{
    m_backing->endOcclusionQuery();
}

void GPURenderPassEncoder::executeBundles(Vector<Ref<GPURenderBundle>>&& bundles)
{
    auto result = WTF::map(bundles, [](auto& bundle) -> Ref<WebGPU::RenderBundle> {
        return bundle->backing();
    });
    m_backing->executeBundles(WTFMove(result));
}

void GPURenderPassEncoder::end()
{
    m_backing->end();
    if (m_device)
        m_backing = m_device->invalidRenderPassEncoder();
}

}
