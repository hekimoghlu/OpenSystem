/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#include "RemoteRenderPassEncoderProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteRenderPassEncoderMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderPassEncoderProxy);

RemoteRenderPassEncoderProxy::RemoteRenderPassEncoderProxy(RemoteCommandEncoderProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_root(parent.root())
{
}

RemoteRenderPassEncoderProxy::~RemoteRenderPassEncoderProxy()
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setPipeline(const WebCore::WebGPU::RenderPipeline& renderPipeline)
{
    auto convertedRenderPipeline = protectedConvertToBackingContext()->convertToBacking(renderPipeline);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetPipeline(convertedRenderPipeline));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setIndexBuffer(const WebCore::WebGPU::Buffer& buffer, WebCore::WebGPU::IndexFormat indexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(buffer);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetIndexBuffer(convertedBuffer, indexFormat, offset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setVertexBuffer(WebCore::WebGPU::Index32 slot, const WebCore::WebGPU::Buffer* buffer, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    if (!buffer) {
        auto sendResult = send(Messages::RemoteRenderPassEncoder::UnsetVertexBuffer(slot, offset, size));
        UNUSED_VARIABLE(sendResult);
        return;
    }

    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(*buffer);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetVertexBuffer(slot, convertedBuffer, offset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::Draw(vertexCount, instanceCount, firstVertex, firstInstance));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::drawIndexed(WebCore::WebGPU::Size32 indexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstIndex,
    std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
    std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::DrawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::drawIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedConvertToBackingContext()->convertToBacking(indirectBuffer);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::DrawIndirect(convertedIndirectBuffer, indirectOffset));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::drawIndexedIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedConvertToBackingContext()->convertToBacking(indirectBuffer);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::DrawIndexedIndirect(convertedIndirectBuffer, indirectOffset));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetBindGroup(index, convertedBindGroup, dynamicOffsets));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::span<const uint32_t> dynamicOffsetsArrayBuffer,
    WebCore::WebGPU::Size64 dynamicOffsetsDataStart,
    WebCore::WebGPU::Size32 dynamicOffsetsDataLength)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetBindGroup(index, convertedBindGroup, Vector<WebCore::WebGPU::BufferDynamicOffset>(dynamicOffsetsArrayBuffer.subspan(dynamicOffsetsDataStart, dynamicOffsetsDataLength))));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::pushDebugGroup(String&& groupLabel)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::PushDebugGroup(WTFMove(groupLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::popDebugGroup()
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::PopDebugGroup());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::insertDebugMarker(String&& markerLabel)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::InsertDebugMarker(WTFMove(markerLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setViewport(float x, float y,
    float width, float height,
    float minDepth, float maxDepth)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetViewport(x, y, width, height, minDepth, maxDepth));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setScissorRect(WebCore::WebGPU::IntegerCoordinate x, WebCore::WebGPU::IntegerCoordinate y,
    WebCore::WebGPU::IntegerCoordinate width, WebCore::WebGPU::IntegerCoordinate height)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetScissorRect(x, y, width, height));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setBlendConstant(WebCore::WebGPU::Color color)
{
    auto convertedColor = protectedConvertToBackingContext()->convertToBacking(color);
    ASSERT(convertedColor);
    if (!convertedColor)
        return;

    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetBlendConstant(*convertedColor));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setStencilReference(WebCore::WebGPU::StencilValue stencilValue)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetStencilReference(stencilValue));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::beginOcclusionQuery(WebCore::WebGPU::Size32 queryIndex)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::BeginOcclusionQuery(queryIndex));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::endOcclusionQuery()
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::EndOcclusionQuery());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::executeBundles(Vector<Ref<WebCore::WebGPU::RenderBundle>>&& renderBundles)
{
    auto convertedRenderBundles = WTF::compactMap(renderBundles, [&](auto& renderBundle) -> std::optional<WebGPUIdentifier> {
        return protectedConvertToBackingContext()->convertToBacking(renderBundle);
    });

    auto sendResult = send(Messages::RemoteRenderPassEncoder::ExecuteBundles(WTFMove(convertedRenderBundles)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::end()
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::End());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderPassEncoderProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteRenderPassEncoder::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

Ref<ConvertToBackingContext> RemoteRenderPassEncoderProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
