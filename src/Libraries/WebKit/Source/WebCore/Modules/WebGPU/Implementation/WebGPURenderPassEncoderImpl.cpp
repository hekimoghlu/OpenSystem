/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#include "WebGPURenderPassEncoderImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBindGroupImpl.h"
#include "WebGPUBufferImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUQuerySetImpl.h"
#include "WebGPURenderBundleImpl.h"
#include "WebGPURenderPipelineImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderPassEncoderImpl);

RenderPassEncoderImpl::RenderPassEncoderImpl(WebGPUPtr<WGPURenderPassEncoder>&& renderPassEncoder, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(renderPassEncoder))
    , m_convertToBackingContext(convertToBackingContext)
{
}

RenderPassEncoderImpl::~RenderPassEncoderImpl() = default;

void RenderPassEncoderImpl::setPipeline(const RenderPipeline& renderPipeline)
{
    wgpuRenderPassEncoderSetPipeline(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(renderPipeline));
}

void RenderPassEncoderImpl::setIndexBuffer(const Buffer& buffer, IndexFormat indexFormat, std::optional<Size64> offset, std::optional<Size64> size)
{
    wgpuRenderPassEncoderSetIndexBuffer(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(buffer), protectedConvertToBackingContext()->convertToBacking(indexFormat), offset.value_or(0), size.value_or(WGPU_WHOLE_SIZE));
}

void RenderPassEncoderImpl::setVertexBuffer(Index32 slot, const Buffer* buffer, std::optional<Size64> offset, std::optional<Size64> size)
{
    wgpuRenderPassEncoderSetVertexBuffer(m_backing.get(), slot, buffer ? protectedConvertToBackingContext()->convertToBacking(*buffer) : nullptr, offset.value_or(0), size.value_or(WGPU_WHOLE_SIZE));
}

void RenderPassEncoderImpl::draw(Size32 vertexCount, std::optional<Size32> instanceCount,
    std::optional<Size32> firstVertex, std::optional<Size32> firstInstance)
{
    wgpuRenderPassEncoderDraw(m_backing.get(), vertexCount, instanceCount.value_or(1), firstVertex.value_or(0), firstInstance.value_or(0));
}

void RenderPassEncoderImpl::drawIndexed(Size32 indexCount, std::optional<Size32> instanceCount,
    std::optional<Size32> firstIndex,
    std::optional<SignedOffset32> baseVertex,
    std::optional<Size32> firstInstance)
{
    wgpuRenderPassEncoderDrawIndexed(m_backing.get(), indexCount, instanceCount.value_or(1), firstIndex.value_or(0), baseVertex.value_or(0), firstInstance.value_or(0));
}

void RenderPassEncoderImpl::drawIndirect(const Buffer& indirectBuffer, Size64 indirectOffset)
{
    wgpuRenderPassEncoderDrawIndirect(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(indirectBuffer), indirectOffset);
}

void RenderPassEncoderImpl::drawIndexedIndirect(const Buffer& indirectBuffer, Size64 indirectOffset)
{
    wgpuRenderPassEncoderDrawIndexedIndirect(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(indirectBuffer), indirectOffset);
}

void RenderPassEncoderImpl::setBindGroup(Index32 index, const BindGroup& bindGroup,
    std::optional<Vector<BufferDynamicOffset>>&& dynamicOffsets)
{
    auto backingOffsets = valueOrDefault(dynamicOffsets);
    wgpuRenderPassEncoderSetBindGroup(m_backing.get(), index, protectedConvertToBackingContext()->convertToBacking(bindGroup), backingOffsets.size(), backingOffsets.data());
}

void RenderPassEncoderImpl::setBindGroup(Index32 index, const BindGroup& bindGroup,
    std::span<const uint32_t> dynamicOffsetsArrayBuffer,
    Size64 dynamicOffsetsDataStart,
    Size32 dynamicOffsetsDataLength)
{
    // FIXME: Use checked algebra.
    wgpuRenderPassEncoderSetBindGroup(m_backing.get(), index, protectedConvertToBackingContext()->convertToBacking(bindGroup), dynamicOffsetsDataLength, dynamicOffsetsArrayBuffer.subspan(dynamicOffsetsDataStart).data());
}

void RenderPassEncoderImpl::pushDebugGroup(String&& groupLabel)
{
    wgpuRenderPassEncoderPushDebugGroup(m_backing.get(), groupLabel.utf8().data());
}

void RenderPassEncoderImpl::popDebugGroup()
{
    wgpuRenderPassEncoderPopDebugGroup(m_backing.get());
}

void RenderPassEncoderImpl::insertDebugMarker(String&& markerLabel)
{
    wgpuRenderPassEncoderInsertDebugMarker(m_backing.get(), markerLabel.utf8().data());
}

void RenderPassEncoderImpl::setViewport(float x, float y,
    float width, float height,
    float minDepth, float maxDepth)
{
    wgpuRenderPassEncoderSetViewport(m_backing.get(), x, y, width, height, minDepth, maxDepth);
}

void RenderPassEncoderImpl::setScissorRect(IntegerCoordinate x, IntegerCoordinate y,
    IntegerCoordinate width, IntegerCoordinate height)
{
    wgpuRenderPassEncoderSetScissorRect(m_backing.get(), x, y, width, height);
}

void RenderPassEncoderImpl::setBlendConstant(Color color)
{
    auto backingColor = protectedConvertToBackingContext()->convertToBacking(color);

    wgpuRenderPassEncoderSetBlendConstant(m_backing.get(), &backingColor);
}

void RenderPassEncoderImpl::setStencilReference(StencilValue stencilValue)
{
    wgpuRenderPassEncoderSetStencilReference(m_backing.get(), stencilValue);
}

void RenderPassEncoderImpl::beginOcclusionQuery(Size32 queryIndex)
{
    wgpuRenderPassEncoderBeginOcclusionQuery(m_backing.get(), queryIndex);
}

void RenderPassEncoderImpl::endOcclusionQuery()
{
    wgpuRenderPassEncoderEndOcclusionQuery(m_backing.get());
}

void RenderPassEncoderImpl::executeBundles(Vector<Ref<RenderBundle>>&& renderBundles)
{
    auto backingBundles = renderBundles.map([&](auto renderBundle) {
        return protectedConvertToBackingContext()->convertToBacking(renderBundle.get());
    });

    wgpuRenderPassEncoderExecuteBundles(m_backing.get(), backingBundles.size(), backingBundles.data());
}

void RenderPassEncoderImpl::end()
{
    wgpuRenderPassEncoderEnd(m_backing.get());
}

void RenderPassEncoderImpl::setLabelInternal(const String& label)
{
    wgpuRenderPassEncoderSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
