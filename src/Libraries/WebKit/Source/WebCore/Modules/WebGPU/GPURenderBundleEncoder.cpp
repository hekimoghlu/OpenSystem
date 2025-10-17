/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#include "GPURenderBundleEncoder.h"

#include "GPUBindGroup.h"
#include "GPUBuffer.h"
#include "GPURenderBundle.h"
#include "GPURenderPipeline.h"

namespace WebCore {

String GPURenderBundleEncoder::label() const
{
    return m_backing->label();
}

void GPURenderBundleEncoder::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

void GPURenderBundleEncoder::setPipeline(const GPURenderPipeline& renderPipeline)
{
    m_backing->setPipeline(renderPipeline.backing());
}

void GPURenderBundleEncoder::setIndexBuffer(const GPUBuffer& buffer, GPUIndexFormat indexFormat, std::optional<GPUSize64> offset, std::optional<GPUSize64> size)
{
    m_backing->setIndexBuffer(buffer.backing(), convertToBacking(indexFormat), offset, size);
}

void GPURenderBundleEncoder::setVertexBuffer(GPUIndex32 slot, const GPUBuffer* buffer, std::optional<GPUSize64> offset, std::optional<GPUSize64> size)
{
    m_backing->setVertexBuffer(slot, buffer ? &buffer->backing() : nullptr, offset, size);
}

void GPURenderBundleEncoder::draw(GPUSize32 vertexCount,
    std::optional<GPUSize32> instanceCount,
    std::optional<GPUSize32> firstVertex, std::optional<GPUSize32> firstInstance)
{
    m_backing->draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void GPURenderBundleEncoder::drawIndexed(GPUSize32 indexCount,
    std::optional<GPUSize32> instanceCount,
    std::optional<GPUSize32> firstIndex,
    std::optional<GPUSignedOffset32> baseVertex,
    std::optional<GPUSize32> firstInstance)
{
    m_backing->drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
}

void GPURenderBundleEncoder::drawIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset)
{
    m_backing->drawIndirect(indirectBuffer.backing(), indirectOffset);
}

void GPURenderBundleEncoder::drawIndexedIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset)
{
    m_backing->drawIndexedIndirect(indirectBuffer.backing(), indirectOffset);
}

void GPURenderBundleEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
    std::optional<Vector<GPUBufferDynamicOffset>>&& dynamicOffsets)
{
    m_backing->setBindGroup(index, bindGroup.backing(), WTFMove(dynamicOffsets));
}

ExceptionOr<void> GPURenderBundleEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
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

void GPURenderBundleEncoder::pushDebugGroup(String&& groupLabel)
{
    m_backing->pushDebugGroup(WTFMove(groupLabel));
}

void GPURenderBundleEncoder::popDebugGroup()
{
    m_backing->popDebugGroup();
}

void GPURenderBundleEncoder::insertDebugMarker(String&& markerLabel)
{
    m_backing->insertDebugMarker(WTFMove(markerLabel));
}

static WebGPU::RenderBundleDescriptor convertToBacking(const std::optional<GPURenderBundleDescriptor>& renderBundleDescriptor)
{
    if (!renderBundleDescriptor)
        return { };
    return renderBundleDescriptor->convertToBacking();
}

ExceptionOr<Ref<GPURenderBundle>> GPURenderBundleEncoder::finish(const std::optional<GPURenderBundleDescriptor>& renderBundleDescriptor)
{
    RefPtr bundle = m_backing->finish(convertToBacking(renderBundleDescriptor));
    if (!bundle)
        return Exception { ExceptionCode::InvalidStateError, "dynamic offsets overflowed"_s };
    return GPURenderBundle::create(bundle.releaseNonNull());
}

}
