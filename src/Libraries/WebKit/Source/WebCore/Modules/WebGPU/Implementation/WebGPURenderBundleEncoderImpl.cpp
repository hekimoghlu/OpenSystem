/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#include "WebGPURenderBundleEncoderImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBindGroupImpl.h"
#include "WebGPUBufferImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPURenderBundleImpl.h"
#include "WebGPURenderPipelineImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderBundleEncoderImpl);

RenderBundleEncoderImpl::RenderBundleEncoderImpl(WebGPUPtr<WGPURenderBundleEncoder>&& renderBundleEncoder, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(renderBundleEncoder))
    , m_convertToBackingContext(convertToBackingContext)
{
}

RenderBundleEncoderImpl::~RenderBundleEncoderImpl() = default;

void RenderBundleEncoderImpl::setPipeline(const RenderPipeline& renderPipeline)
{
    wgpuRenderBundleEncoderSetPipeline(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(renderPipeline));
}

void RenderBundleEncoderImpl::setIndexBuffer(const Buffer& buffer, IndexFormat indexFormat, std::optional<Size64> offset, std::optional<Size64> size)
{
    wgpuRenderBundleEncoderSetIndexBuffer(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(buffer), protectedConvertToBackingContext()->convertToBacking(indexFormat), offset.value_or(0), size.value_or(WGPU_WHOLE_SIZE));
}

void RenderBundleEncoderImpl::setVertexBuffer(Index32 slot, const Buffer* buffer, std::optional<Size64> offset, std::optional<Size64> size)
{
    wgpuRenderBundleEncoderSetVertexBuffer(m_backing.get(), slot, buffer ? protectedConvertToBackingContext()->convertToBacking(*buffer) : nullptr, offset.value_or(0), size.value_or(WGPU_WHOLE_SIZE));
}

void RenderBundleEncoderImpl::draw(Size32 vertexCount, std::optional<Size32> instanceCount,
    std::optional<Size32> firstVertex, std::optional<Size32> firstInstance)
{
    wgpuRenderBundleEncoderDraw(m_backing.get(), vertexCount, instanceCount.value_or(1), firstVertex.value_or(0), firstInstance.value_or(0));
}

void RenderBundleEncoderImpl::drawIndexed(Size32 indexCount, std::optional<Size32> instanceCount,
    std::optional<Size32> firstIndex,
    std::optional<SignedOffset32> baseVertex,
    std::optional<Size32> firstInstance)
{
    wgpuRenderBundleEncoderDrawIndexed(m_backing.get(), indexCount, instanceCount.value_or(1), firstIndex.value_or(0), baseVertex.value_or(0), firstInstance.value_or(0));
}

void RenderBundleEncoderImpl::drawIndirect(const Buffer& indirectBuffer, Size64 indirectOffset)
{
    wgpuRenderBundleEncoderDrawIndirect(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(indirectBuffer), indirectOffset);
}

void RenderBundleEncoderImpl::drawIndexedIndirect(const Buffer& indirectBuffer, Size64 indirectOffset)
{
    wgpuRenderBundleEncoderDrawIndexedIndirect(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(indirectBuffer), indirectOffset);
}

void RenderBundleEncoderImpl::setBindGroup(Index32 index, const BindGroup& bindGroup,
    std::optional<Vector<BufferDynamicOffset>>&& dynamicOffsets)
{
    auto backingOffsets = valueOrDefault(dynamicOffsets);
    wgpuRenderBundleEncoderSetBindGroupWithDynamicOffsets(m_backing.get(), index, protectedConvertToBackingContext()->convertToBacking(bindGroup), WTFMove(dynamicOffsets));
}

void RenderBundleEncoderImpl::setBindGroup(Index32, const BindGroup&,
    std::span<const uint32_t>,
    Size64,
    Size32)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void RenderBundleEncoderImpl::pushDebugGroup(String&& groupLabel)
{
    wgpuRenderBundleEncoderPushDebugGroup(m_backing.get(), groupLabel.utf8().data());
}

void RenderBundleEncoderImpl::popDebugGroup()
{
    wgpuRenderBundleEncoderPopDebugGroup(m_backing.get());
}

void RenderBundleEncoderImpl::insertDebugMarker(String&& markerLabel)
{
    wgpuRenderBundleEncoderInsertDebugMarker(m_backing.get(), markerLabel.utf8().data());
}

RefPtr<RenderBundle> RenderBundleEncoderImpl::finish(const RenderBundleDescriptor& descriptor)
{
    auto label = descriptor.label.utf8();

    WGPURenderBundleDescriptor backingDescriptor {
        nullptr,
        label.data(),
    };

    return RenderBundleImpl::create(adoptWebGPU(wgpuRenderBundleEncoderFinish(m_backing.get(), &backingDescriptor)), m_convertToBackingContext);
}

void RenderBundleEncoderImpl::setLabelInternal(const String& label)
{
    wgpuRenderBundleEncoderSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
