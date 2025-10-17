/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "WebGPUComputePassEncoderImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBindGroupImpl.h"
#include "WebGPUBufferImpl.h"
#include "WebGPUComputePipelineImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUQuerySetImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ComputePassEncoderImpl);

ComputePassEncoderImpl::ComputePassEncoderImpl(WebGPUPtr<WGPUComputePassEncoder>&& computePassEncoder, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(computePassEncoder))
    , m_convertToBackingContext(convertToBackingContext)
{
}

ComputePassEncoderImpl::~ComputePassEncoderImpl() = default;

void ComputePassEncoderImpl::setPipeline(const ComputePipeline& computePipeline)
{
    wgpuComputePassEncoderSetPipeline(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(computePipeline));
}

void ComputePassEncoderImpl::dispatch(Size32 workgroupCountX, Size32 workgroupCountY, Size32 workgroupCountZ)
{
    wgpuComputePassEncoderDispatchWorkgroups(m_backing.get(), workgroupCountX, workgroupCountY, workgroupCountZ);
}

void ComputePassEncoderImpl::dispatchIndirect(const Buffer& indirectBuffer, Size64 indirectOffset)
{
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(indirectBuffer), indirectOffset);
}

void ComputePassEncoderImpl::end()
{
    wgpuComputePassEncoderEnd(m_backing.get());
}

void ComputePassEncoderImpl::setBindGroup(Index32 index, const BindGroup& bindGroup,
    std::optional<Vector<BufferDynamicOffset>>&& offsets)
{
    auto backingOffsets = valueOrDefault(offsets);
    wgpuComputePassEncoderSetBindGroup(m_backing.get(), index, protectedConvertToBackingContext()->convertToBacking(bindGroup), static_cast<uint32_t>(backingOffsets.size()), backingOffsets.data());
}

void ComputePassEncoderImpl::setBindGroup(Index32 index, const BindGroup& bindGroup,
    std::span<const uint32_t> dynamicOffsetsArrayBuffer,
    Size64 dynamicOffsetsDataStart,
    Size32 dynamicOffsetsDataLength)
{
    wgpuComputePassEncoderSetBindGroup(m_backing.get(), index, protectedConvertToBackingContext()->convertToBacking(bindGroup), dynamicOffsetsDataLength, dynamicOffsetsArrayBuffer.subspan(dynamicOffsetsDataStart).data());
}

void ComputePassEncoderImpl::pushDebugGroup(String&& groupLabel)
{
    wgpuComputePassEncoderPushDebugGroup(m_backing.get(), groupLabel.utf8().data());
}

void ComputePassEncoderImpl::popDebugGroup()
{
    wgpuComputePassEncoderPopDebugGroup(m_backing.get());
}

void ComputePassEncoderImpl::insertDebugMarker(String&& markerLabel)
{
    wgpuComputePassEncoderInsertDebugMarker(m_backing.get(), markerLabel.utf8().data());
}

void ComputePassEncoderImpl::setLabelInternal(const String& label)
{
    wgpuComputePassEncoderSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
