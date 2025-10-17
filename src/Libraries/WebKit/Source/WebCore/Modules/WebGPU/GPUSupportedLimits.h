/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

#include "WebGPUSupportedLimits.h"

namespace WebCore {

class GPUSupportedLimits : public RefCounted<GPUSupportedLimits> {
public:
    static Ref<GPUSupportedLimits> create(Ref<WebGPU::SupportedLimits>&& backing)
    {
        return adoptRef(*new GPUSupportedLimits(WTFMove(backing)));
    }

    uint32_t maxTextureDimension1D() const;
    uint32_t maxTextureDimension2D() const;
    uint32_t maxTextureDimension3D() const;
    uint32_t maxTextureArrayLayers() const;
    uint32_t maxBindGroups() const;
    uint32_t maxBindGroupsPlusVertexBuffers() const;
    uint32_t maxBindingsPerBindGroup() const;
    uint32_t maxDynamicUniformBuffersPerPipelineLayout() const;
    uint32_t maxDynamicStorageBuffersPerPipelineLayout() const;
    uint32_t maxSampledTexturesPerShaderStage() const;
    uint32_t maxSamplersPerShaderStage() const;
    uint32_t maxStorageBuffersPerShaderStage() const;
    uint32_t maxStorageTexturesPerShaderStage() const;
    uint32_t maxUniformBuffersPerShaderStage() const;
    uint64_t maxUniformBufferBindingSize() const;
    uint64_t maxStorageBufferBindingSize() const;
    uint32_t minUniformBufferOffsetAlignment() const;
    uint32_t minStorageBufferOffsetAlignment() const;
    uint32_t maxVertexBuffers() const;
    uint64_t maxBufferSize() const;
    uint32_t maxVertexAttributes() const;
    uint32_t maxVertexBufferArrayStride() const;
    uint32_t maxInterStageShaderComponents() const;
    uint32_t maxInterStageShaderVariables() const;
    uint32_t maxColorAttachments() const;
    uint32_t maxColorAttachmentBytesPerSample() const;
    uint32_t maxComputeWorkgroupStorageSize() const;
    uint32_t maxComputeInvocationsPerWorkgroup() const;
    uint32_t maxComputeWorkgroupSizeX() const;
    uint32_t maxComputeWorkgroupSizeY() const;
    uint32_t maxComputeWorkgroupSizeZ() const;
    uint32_t maxComputeWorkgroupsPerDimension() const;
    uint32_t maxStorageBuffersInFragmentStage() const;
    uint32_t maxStorageTexturesInFragmentStage() const;

    WebGPU::SupportedLimits& backing() { return m_backing; }
    const WebGPU::SupportedLimits& backing() const { return m_backing; }

private:
    GPUSupportedLimits(Ref<WebGPU::SupportedLimits>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::SupportedLimits> m_backing;
};

}
