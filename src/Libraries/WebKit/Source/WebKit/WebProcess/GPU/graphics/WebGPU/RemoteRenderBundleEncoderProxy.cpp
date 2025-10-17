/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#include "RemoteRenderBundleEncoderProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteRenderBundleEncoderMessages.h"
#include "RemoteRenderBundleProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRenderBundleEncoderProxy);

RemoteRenderBundleEncoderProxy::RemoteRenderBundleEncoderProxy(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemoteRenderBundleEncoderProxy::~RemoteRenderBundleEncoderProxy()
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::setPipeline(const WebCore::WebGPU::RenderPipeline& renderPipeline)
{
    auto convertedRenderPipeline = protectedConvertToBackingContext()->convertToBacking(renderPipeline);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetPipeline(convertedRenderPipeline));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::setIndexBuffer(const WebCore::WebGPU::Buffer& buffer, WebCore::WebGPU::IndexFormat indexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(buffer);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetIndexBuffer(convertedBuffer, indexFormat, offset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::setVertexBuffer(WebCore::WebGPU::Index32 slot, const WebCore::WebGPU::Buffer* buffer, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64> size)
{
    if (!buffer) {
        auto sendResult = send(Messages::RemoteRenderBundleEncoder::UnsetVertexBuffer(slot, offset, size));
        UNUSED_VARIABLE(sendResult);
        return;
    }
    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(*buffer);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetVertexBuffer(slot, convertedBuffer, offset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::Draw(vertexCount, instanceCount, firstVertex, firstInstance));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::drawIndexed(WebCore::WebGPU::Size32 indexCount,
    std::optional<WebCore::WebGPU::Size32> instanceCount,
    std::optional<WebCore::WebGPU::Size32> firstIndex,
    std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
    std::optional<WebCore::WebGPU::Size32> firstInstance)
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::DrawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::drawIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedConvertToBackingContext()->convertToBacking(indirectBuffer);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::DrawIndirect(convertedIndirectBuffer, indirectOffset));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::drawIndexedIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedConvertToBackingContext()->convertToBacking(indirectBuffer);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::DrawIndexedIndirect(convertedIndirectBuffer, indirectOffset));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetBindGroup(index, convertedBindGroup, dynamicOffsets));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::span<const uint32_t> dynamicOffsetsArrayBuffer,
    WebCore::WebGPU::Size64 dynamicOffsetsDataStart,
    WebCore::WebGPU::Size32 dynamicOffsetsDataLength)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetBindGroup(index, convertedBindGroup, Vector<WebCore::WebGPU::BufferDynamicOffset>(dynamicOffsetsArrayBuffer.subspan(dynamicOffsetsDataStart, dynamicOffsetsDataLength))));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::pushDebugGroup(String&& groupLabel)
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::PushDebugGroup(WTFMove(groupLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::popDebugGroup()
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::PopDebugGroup());
    UNUSED_VARIABLE(sendResult);
}

void RemoteRenderBundleEncoderProxy::insertDebugMarker(String&& markerLabel)
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::InsertDebugMarker(WTFMove(markerLabel)));
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::RenderBundle> RemoteRenderBundleEncoderProxy::finish(const WebCore::WebGPU::RenderBundleDescriptor& descriptor)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedDescriptor = convertToBackingContext->convertToBacking(descriptor);
    if (!convertedDescriptor)
        return nullptr;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::Finish(*convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    return RemoteRenderBundleProxy::create(m_parent, convertToBackingContext, identifier);
}

void RemoteRenderBundleEncoderProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteRenderBundleEncoder::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

Ref<ConvertToBackingContext> RemoteRenderBundleEncoderProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
