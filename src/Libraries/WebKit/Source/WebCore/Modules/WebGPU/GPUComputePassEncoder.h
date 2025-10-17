/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#pragma once

#include "ExceptionOr.h"
#include "GPUIntegralTypes.h"
#include "WebGPUComputePassEncoder.h"
#include <JavaScriptCore/Uint32Array.h>
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBindGroup;
class GPUBuffer;
class GPUComputePipeline;
class GPUQuerySet;

namespace WebGPU {
class Device;
}

class GPUComputePassEncoder : public RefCounted<GPUComputePassEncoder> {
public:
    static Ref<GPUComputePassEncoder> create(Ref<WebGPU::ComputePassEncoder>&& backing, WebGPU::Device& device)
    {
        return adoptRef(*new GPUComputePassEncoder(WTFMove(backing), device));
    }

    String label() const;
    void setLabel(String&&);

    void setPipeline(const GPUComputePipeline&);
    void dispatchWorkgroups(GPUSize32 workgroupCountX, std::optional<GPUSize32> workgroupCountY, std::optional<GPUSize32> workgroupCountZ);
    void dispatchWorkgroupsIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset);

    void end();

    void setBindGroup(GPUIndex32, const GPUBindGroup&,
        std::optional<Vector<GPUBufferDynamicOffset>>&&);

    ExceptionOr<void> setBindGroup(GPUIndex32, const GPUBindGroup&,
        const JSC::Uint32Array& dynamicOffsetsData,
        GPUSize64 dynamicOffsetsDataStart,
        GPUSize32 dynamicOffsetsDataLength);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    WebGPU::ComputePassEncoder& backing() { return m_backing; }
    const WebGPU::ComputePassEncoder& backing() const { return m_backing; }

private:
    GPUComputePassEncoder(Ref<WebGPU::ComputePassEncoder>&&, WebGPU::Device&);

    Ref<WebGPU::ComputePassEncoder> m_backing;
    WeakPtr<WebGPU::Device> m_device;
};

}
