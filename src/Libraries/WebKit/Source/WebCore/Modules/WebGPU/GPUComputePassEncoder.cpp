/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include "GPUComputePassEncoder.h"

#include "GPUBindGroup.h"
#include "GPUBuffer.h"
#include "GPUComputePipeline.h"
#include "GPUQuerySet.h"
#include "WebGPUDevice.h"

namespace WebCore {

GPUComputePassEncoder::GPUComputePassEncoder(Ref<WebGPU::ComputePassEncoder>&& backing, WebGPU::Device& device)
    : m_backing(WTFMove(backing))
    , m_device(&device)
{
}

String GPUComputePassEncoder::label() const
{
    return m_backing->label();
}

void GPUComputePassEncoder::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

void GPUComputePassEncoder::setPipeline(const GPUComputePipeline& computePipeline)
{
    m_backing->setPipeline(computePipeline.backing());
}

void GPUComputePassEncoder::dispatchWorkgroups(GPUSize32 workgroupCountX, std::optional<GPUSize32> workgroupCountY, std::optional<GPUSize32> workgroupCountZ)
{
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=240219 we should be able to specify the
    // default values via the idl file
    m_backing->dispatch(workgroupCountX, workgroupCountY.value_or(1), workgroupCountZ.value_or(1));
}

void GPUComputePassEncoder::dispatchWorkgroupsIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset)
{
    m_backing->dispatchIndirect(indirectBuffer.backing(), indirectOffset);
}

void GPUComputePassEncoder::end()
{
    m_backing->end();
    if (m_device)
        m_backing = m_device->invalidComputePassEncoder();
}

void GPUComputePassEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
    std::optional<Vector<GPUBufferDynamicOffset>>&& dynamicOffsets)
{
    m_backing->setBindGroup(index, bindGroup.backing(), WTFMove(dynamicOffsets));
}

ExceptionOr<void> GPUComputePassEncoder::setBindGroup(GPUIndex32 index, const GPUBindGroup& bindGroup,
    const JSC::Uint32Array& dynamicOffsetsData,
    GPUSize64 dynamicOffsetsDataStart,
    GPUSize32 dynamicOffsetsDataLength)
{
    auto offset = checkedSum<uint64_t>(dynamicOffsetsDataStart, dynamicOffsetsDataLength);
    if (offset.hasOverflowed() || offset > dynamicOffsetsData.length())
        return Exception { ExceptionCode::RangeError, "dynamic offsets overflowed"_s };

    m_backing->setBindGroup(index, bindGroup.backing(), dynamicOffsetsData.typedSpan(), dynamicOffsetsDataStart, dynamicOffsetsDataLength);
    return { };
}

void GPUComputePassEncoder::pushDebugGroup(String&& groupLabel)
{
    m_backing->pushDebugGroup(WTFMove(groupLabel));
}

void GPUComputePassEncoder::popDebugGroup()
{
    m_backing->popDebugGroup();
}

void GPUComputePassEncoder::insertDebugMarker(String&& markerLabel)
{
    m_backing->insertDebugMarker(WTFMove(markerLabel));
}

}
