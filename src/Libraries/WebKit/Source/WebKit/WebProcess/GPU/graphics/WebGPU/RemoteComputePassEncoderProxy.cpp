/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "RemoteComputePassEncoderProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteComputePassEncoderMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteComputePassEncoderProxy);

RemoteComputePassEncoderProxy::RemoteComputePassEncoderProxy(RemoteCommandEncoderProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_root(parent.root())
{
}

RemoteComputePassEncoderProxy::~RemoteComputePassEncoderProxy()
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::setPipeline(const WebCore::WebGPU::ComputePipeline& computePipeline)
{
    auto convertedComputePipeline = protectedConvertToBackingContext()->convertToBacking(computePipeline);

    auto sendResult = send(Messages::RemoteComputePassEncoder::SetPipeline(convertedComputePipeline));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::dispatch(WebCore::WebGPU::Size32 workgroupCountX, WebCore::WebGPU::Size32 workgroupCountY, WebCore::WebGPU::Size32 workgroupCountZ)
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::Dispatch(workgroupCountX, workgroupCountY, workgroupCountZ));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::dispatchIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset)
{
    auto convertedIndirectBuffer = protectedConvertToBackingContext()->convertToBacking(indirectBuffer);

    auto sendResult = send(Messages::RemoteComputePassEncoder::DispatchIndirect(convertedIndirectBuffer, indirectOffset));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::end()
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::End());
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& offsets)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteComputePassEncoder::SetBindGroup(index, convertedBindGroup, WTFMove(offsets)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::setBindGroup(WebCore::WebGPU::Index32 index, const WebCore::WebGPU::BindGroup& bindGroup,
    std::span<const uint32_t> dynamicOffsetsArrayBuffer,
    WebCore::WebGPU::Size64 dynamicOffsetsDataStart,
    WebCore::WebGPU::Size32 dynamicOffsetsDataLength)
{
    auto convertedBindGroup = protectedConvertToBackingContext()->convertToBacking(bindGroup);

    auto sendResult = send(Messages::RemoteComputePassEncoder::SetBindGroup(index, convertedBindGroup, Vector<WebCore::WebGPU::BufferDynamicOffset>(dynamicOffsetsArrayBuffer.subspan(dynamicOffsetsDataStart, dynamicOffsetsDataLength))));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::pushDebugGroup(String&& groupLabel)
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::PushDebugGroup(WTFMove(groupLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::popDebugGroup()
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::PopDebugGroup());
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::insertDebugMarker(String&& markerLabel)
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::InsertDebugMarker(WTFMove(markerLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteComputePassEncoderProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteComputePassEncoder::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
