/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

#include <cstdint>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore::WebGPU {

class SupportedLimits final : public RefCounted<SupportedLimits> {
public:
    static Ref<SupportedLimits> create(
        uint32_t maxTextureDimension1D,
        uint32_t maxTextureDimension2D,
        uint32_t maxTextureDimension3D,
        uint32_t maxTextureArrayLayers,
        uint32_t maxBindGroups,
        uint32_t maxBindGroupsPlusVertexBuffers,
        uint32_t maxBindingsPerBindGroup,
        uint32_t maxDynamicUniformBuffersPerPipelineLayout,
        uint32_t maxDynamicStorageBuffersPerPipelineLayout,
        uint32_t maxSampledTexturesPerShaderStage,
        uint32_t maxSamplersPerShaderStage,
        uint32_t maxStorageBuffersPerShaderStage,
        uint32_t maxStorageTexturesPerShaderStage,
        uint32_t maxUniformBuffersPerShaderStage,
        uint64_t maxUniformBufferBindingSize,
        uint64_t maxStorageBufferBindingSize,
        uint32_t minUniformBufferOffsetAlignment,
        uint32_t minStorageBufferOffsetAlignment,
        uint32_t maxVertexBuffers,
        uint64_t maxBufferSize,
        uint32_t maxVertexAttributes,
        uint32_t maxVertexBufferArrayStride,
        uint32_t maxInterStageShaderComponents,
        uint32_t maxInterStageShaderVariables,
        uint32_t maxColorAttachments,
        uint32_t maxColorAttachmentBytesPerSample,
        uint32_t maxComputeWorkgroupStorageSize,
        uint32_t maxComputeInvocationsPerWorkgroup,
        uint32_t maxComputeWorkgroupSizeX,
        uint32_t maxComputeWorkgroupSizeY,
        uint32_t maxComputeWorkgroupSizeZ,
        uint32_t maxComputeWorkgroupsPerDimension,
        uint32_t maxStorageBuffersInFragmentStage,
        uint32_t maxStorageTexturesInFragmentStage)
    {
        return adoptRef(*new SupportedLimits(
            maxTextureDimension1D,
            maxTextureDimension2D,
            maxTextureDimension3D,
            maxTextureArrayLayers,
            maxBindGroups,
            maxBindGroupsPlusVertexBuffers,
            maxBindingsPerBindGroup,
            maxDynamicUniformBuffersPerPipelineLayout,
            maxDynamicStorageBuffersPerPipelineLayout,
            maxSampledTexturesPerShaderStage,
            maxSamplersPerShaderStage,
            maxStorageBuffersPerShaderStage,
            maxStorageTexturesPerShaderStage,
            maxUniformBuffersPerShaderStage,
            maxUniformBufferBindingSize,
            maxStorageBufferBindingSize,
            minUniformBufferOffsetAlignment,
            minStorageBufferOffsetAlignment,
            maxVertexBuffers,
            maxBufferSize,
            maxVertexAttributes,
            maxVertexBufferArrayStride,
            maxInterStageShaderComponents,
            maxInterStageShaderVariables,
            maxColorAttachments,
            maxColorAttachmentBytesPerSample,
            maxComputeWorkgroupStorageSize,
            maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupsPerDimension,
            maxStorageBuffersInFragmentStage,
            maxStorageTexturesInFragmentStage));
    }

    static Ref<SupportedLimits> clone(const SupportedLimits& limits)
    {
        return adoptRef(*new SupportedLimits(
            limits.maxTextureDimension1D(),
            limits.maxTextureDimension2D(),
            limits.maxTextureDimension3D(),
            limits.maxTextureArrayLayers(),
            limits.maxBindGroups(),
            limits.maxBindGroupsPlusVertexBuffers(),
            limits.maxBindingsPerBindGroup(),
            limits.maxDynamicUniformBuffersPerPipelineLayout(),
            limits.maxDynamicStorageBuffersPerPipelineLayout(),
            limits.maxSampledTexturesPerShaderStage(),
            limits.maxSamplersPerShaderStage(),
            limits.maxStorageBuffersPerShaderStage(),
            limits.maxStorageTexturesPerShaderStage(),
            limits.maxUniformBuffersPerShaderStage(),
            limits.maxUniformBufferBindingSize(),
            limits.maxStorageBufferBindingSize(),
            limits.minUniformBufferOffsetAlignment(),
            limits.minStorageBufferOffsetAlignment(),
            limits.maxVertexBuffers(),
            limits.maxBufferSize(),
            limits.maxVertexAttributes(),
            limits.maxVertexBufferArrayStride(),
            limits.maxInterStageShaderComponents(),
            limits.maxInterStageShaderVariables(),
            limits.maxColorAttachments(),
            limits.maxColorAttachmentBytesPerSample(),
            limits.maxComputeWorkgroupStorageSize(),
            limits.maxComputeInvocationsPerWorkgroup(),
            limits.maxComputeWorkgroupSizeX(),
            limits.maxComputeWorkgroupSizeY(),
            limits.maxComputeWorkgroupSizeZ(),
            limits.maxComputeWorkgroupsPerDimension(),
            limits.maxStorageBuffersInFragmentStage(),
            limits.maxStorageTexturesInFragmentStage()));
    }

    uint32_t maxTextureDimension1D() const { return m_maxTextureDimension1D; }
    uint32_t maxTextureDimension2D() const { return m_maxTextureDimension2D; }
    uint32_t maxTextureDimension3D() const { return m_maxTextureDimension3D; }
    uint32_t maxTextureArrayLayers() const { return m_maxTextureArrayLayers; }
    uint32_t maxBindGroups() const { return m_maxBindGroups; }
    uint32_t maxBindGroupsPlusVertexBuffers() const { return m_maxBindGroupsPlusVertexBuffers; }
    uint32_t maxBindingsPerBindGroup() const { return m_maxBindingsPerBindGroup; }
    uint32_t maxDynamicUniformBuffersPerPipelineLayout() const { return m_maxDynamicUniformBuffersPerPipelineLayout; }
    uint32_t maxDynamicStorageBuffersPerPipelineLayout() const { return m_maxDynamicStorageBuffersPerPipelineLayout; }
    uint32_t maxSampledTexturesPerShaderStage() const { return m_maxSampledTexturesPerShaderStage; }
    uint32_t maxSamplersPerShaderStage() const { return m_maxSamplersPerShaderStage; }
    uint32_t maxStorageBuffersPerShaderStage() const { return m_maxStorageBuffersPerShaderStage; }
    uint32_t maxStorageTexturesPerShaderStage() const { return m_maxStorageTexturesPerShaderStage; }
    uint32_t maxUniformBuffersPerShaderStage() const { return m_maxUniformBuffersPerShaderStage; }
    uint64_t maxUniformBufferBindingSize() const { return m_maxUniformBufferBindingSize; }
    uint64_t maxStorageBufferBindingSize() const { return m_maxStorageBufferBindingSize; }
    uint32_t minUniformBufferOffsetAlignment() const { return m_minUniformBufferOffsetAlignment; }
    uint32_t minStorageBufferOffsetAlignment() const { return m_minStorageBufferOffsetAlignment; }
    uint32_t maxVertexBuffers() const { return m_maxVertexBuffers; }
    uint64_t maxBufferSize() const { return m_maxBufferSize; }
    uint32_t maxVertexAttributes() const { return m_maxVertexAttributes; }
    uint32_t maxVertexBufferArrayStride() const { return m_maxVertexBufferArrayStride; }
    uint32_t maxInterStageShaderComponents() const { return m_maxInterStageShaderComponents; }
    uint32_t maxInterStageShaderVariables() const { return m_maxInterStageShaderVariables; }
    uint32_t maxColorAttachments() const { return m_maxColorAttachments; }
    uint32_t maxColorAttachmentBytesPerSample() const { return m_maxColorAttachmentBytesPerSample; }
    uint32_t maxComputeWorkgroupStorageSize() const { return m_maxComputeWorkgroupStorageSize; }
    uint32_t maxComputeInvocationsPerWorkgroup() const { return m_maxComputeInvocationsPerWorkgroup; }
    uint32_t maxComputeWorkgroupSizeX() const { return m_maxComputeWorkgroupSizeX; }
    uint32_t maxComputeWorkgroupSizeY() const { return m_maxComputeWorkgroupSizeY; }
    uint32_t maxComputeWorkgroupSizeZ() const { return m_maxComputeWorkgroupSizeZ; }
    uint32_t maxComputeWorkgroupsPerDimension() const { return m_maxComputeWorkgroupsPerDimension; }
    uint32_t maxStorageBuffersInFragmentStage() const { return m_maxStorageBuffersInFragmentStage; }
    uint32_t maxStorageTexturesInFragmentStage() const { return m_maxStorageTexturesInFragmentStage; }

private:
    SupportedLimits(
        uint32_t maxTextureDimension1D,
        uint32_t maxTextureDimension2D,
        uint32_t maxTextureDimension3D,
        uint32_t maxTextureArrayLayers,
        uint32_t maxBindGroups,
        uint32_t maxBindGroupsPlusVertexBuffers,
        uint32_t maxBindingsPerBindGroup,
        uint32_t maxDynamicUniformBuffersPerPipelineLayout,
        uint32_t maxDynamicStorageBuffersPerPipelineLayout,
        uint32_t maxSampledTexturesPerShaderStage,
        uint32_t maxSamplersPerShaderStage,
        uint32_t maxStorageBuffersPerShaderStage,
        uint32_t maxStorageTexturesPerShaderStage,
        uint32_t maxUniformBuffersPerShaderStage,
        uint64_t maxUniformBufferBindingSize,
        uint64_t maxStorageBufferBindingSize,
        uint32_t minUniformBufferOffsetAlignment,
        uint32_t minStorageBufferOffsetAlignment,
        uint32_t maxVertexBuffers,
        uint64_t maxBufferSize,
        uint32_t maxVertexAttributes,
        uint32_t maxVertexBufferArrayStride,
        uint32_t maxInterStageShaderComponents,
        uint32_t maxInterStageShaderVariables,
        uint32_t maxColorAttachments,
        uint32_t maxColorAttachmentBytesPerSample,
        uint32_t maxComputeWorkgroupStorageSize,
        uint32_t maxComputeInvocationsPerWorkgroup,
        uint32_t maxComputeWorkgroupSizeX,
        uint32_t maxComputeWorkgroupSizeY,
        uint32_t maxComputeWorkgroupSizeZ,
        uint32_t maxComputeWorkgroupsPerDimension,
        uint32_t maxStorageBuffersInFragmentStage,
        uint32_t maxStorageTexturesInFragmentStage)
            : m_maxTextureDimension1D(maxTextureDimension1D)
            , m_maxTextureDimension2D(maxTextureDimension2D)
            , m_maxTextureDimension3D(maxTextureDimension3D)
            , m_maxTextureArrayLayers(maxTextureArrayLayers)
            , m_maxBindGroups(maxBindGroups)
            , m_maxBindGroupsPlusVertexBuffers(maxBindGroupsPlusVertexBuffers)
            , m_maxBindingsPerBindGroup(maxBindingsPerBindGroup)
            , m_maxDynamicUniformBuffersPerPipelineLayout(maxDynamicUniformBuffersPerPipelineLayout)
            , m_maxDynamicStorageBuffersPerPipelineLayout(maxDynamicStorageBuffersPerPipelineLayout)
            , m_maxSampledTexturesPerShaderStage(maxSampledTexturesPerShaderStage)
            , m_maxSamplersPerShaderStage(maxSamplersPerShaderStage)
            , m_maxStorageBuffersPerShaderStage(maxStorageBuffersPerShaderStage)
            , m_maxStorageTexturesPerShaderStage(maxStorageTexturesPerShaderStage)
            , m_maxUniformBuffersPerShaderStage(maxUniformBuffersPerShaderStage)
            , m_maxUniformBufferBindingSize(maxUniformBufferBindingSize)
            , m_maxStorageBufferBindingSize(maxStorageBufferBindingSize)
            , m_minUniformBufferOffsetAlignment(minUniformBufferOffsetAlignment)
            , m_minStorageBufferOffsetAlignment(minStorageBufferOffsetAlignment)
            , m_maxVertexBuffers(maxVertexBuffers)
            , m_maxBufferSize(maxBufferSize)
            , m_maxVertexAttributes(maxVertexAttributes)
            , m_maxVertexBufferArrayStride(maxVertexBufferArrayStride)
            , m_maxInterStageShaderComponents(maxInterStageShaderComponents)
            , m_maxInterStageShaderVariables(maxInterStageShaderVariables)
            , m_maxColorAttachments(maxColorAttachments)
            , m_maxColorAttachmentBytesPerSample(maxColorAttachmentBytesPerSample)
            , m_maxComputeWorkgroupStorageSize(maxComputeWorkgroupStorageSize)
            , m_maxComputeInvocationsPerWorkgroup(maxComputeInvocationsPerWorkgroup)
            , m_maxComputeWorkgroupSizeX(maxComputeWorkgroupSizeX)
            , m_maxComputeWorkgroupSizeY(maxComputeWorkgroupSizeY)
            , m_maxComputeWorkgroupSizeZ(maxComputeWorkgroupSizeZ)
            , m_maxComputeWorkgroupsPerDimension(maxComputeWorkgroupsPerDimension)
            , m_maxStorageBuffersInFragmentStage(maxStorageBuffersInFragmentStage)
            , m_maxStorageTexturesInFragmentStage(maxStorageTexturesInFragmentStage)
    {
    }

    SupportedLimits(const SupportedLimits&) = delete;
    SupportedLimits(SupportedLimits&&) = delete;
    SupportedLimits& operator=(const SupportedLimits&) = delete;
    SupportedLimits& operator=(SupportedLimits&&) = delete;

    uint32_t m_maxTextureDimension1D { 0 };
    uint32_t m_maxTextureDimension2D { 0 };
    uint32_t m_maxTextureDimension3D { 0 };
    uint32_t m_maxTextureArrayLayers { 0 };
    uint32_t m_maxBindGroups { 0 };
    uint32_t m_maxBindGroupsPlusVertexBuffers { 0 };
    uint32_t m_maxBindingsPerBindGroup { 0 };
    uint32_t m_maxDynamicUniformBuffersPerPipelineLayout { 0 };
    uint32_t m_maxDynamicStorageBuffersPerPipelineLayout { 0 };
    uint32_t m_maxSampledTexturesPerShaderStage { 0 };
    uint32_t m_maxSamplersPerShaderStage { 0 };
    uint32_t m_maxStorageBuffersPerShaderStage { 0 };
    uint32_t m_maxStorageTexturesPerShaderStage { 0 };
    uint32_t m_maxUniformBuffersPerShaderStage { 0 };
    uint64_t m_maxUniformBufferBindingSize { 0 };
    uint64_t m_maxStorageBufferBindingSize { 0 };
    uint32_t m_minUniformBufferOffsetAlignment { 0 };
    uint32_t m_minStorageBufferOffsetAlignment { 0 };
    uint32_t m_maxVertexBuffers { 0 };
    uint64_t m_maxBufferSize { 0 };
    uint32_t m_maxVertexAttributes { 0 };
    uint32_t m_maxVertexBufferArrayStride { 0 };
    uint32_t m_maxInterStageShaderComponents { 0 };
    uint32_t m_maxInterStageShaderVariables { 0 };
    uint32_t m_maxColorAttachments { 0 };
    uint32_t m_maxColorAttachmentBytesPerSample { 0 };
    uint32_t m_maxComputeWorkgroupStorageSize { 0 };
    uint32_t m_maxComputeInvocationsPerWorkgroup { 0 };
    uint32_t m_maxComputeWorkgroupSizeX { 0 };
    uint32_t m_maxComputeWorkgroupSizeY { 0 };
    uint32_t m_maxComputeWorkgroupSizeZ { 0 };
    uint32_t m_maxComputeWorkgroupsPerDimension { 0 };
    uint32_t m_maxStorageBuffersInFragmentStage { 0 };
    uint32_t m_maxStorageTexturesInFragmentStage { 0 };
};

} // namespace WebCore::WebGPU
